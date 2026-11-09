package org.futo.inputmethod.latin.replay

import android.content.Context
import android.content.res.Configuration
import android.os.SystemClock
import android.text.InputType
import android.view.MotionEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import androidx.datastore.preferences.core.booleanPreferencesKey
import org.futo.inputmethod.event.Event
import org.futo.inputmethod.keyboard.Keyboard
import org.futo.inputmethod.latin.BuildConfig
import org.futo.inputmethod.latin.SuggestedWords
import org.futo.inputmethod.latin.settings.Settings
import org.futo.inputmethod.latin.uix.SettingsKey
import org.futo.inputmethod.latin.uix.getSettingBlocking
import org.futo.inputmethod.latin.uix.isDirectBootUnlocked
import org.futo.inputmethod.latin.utils.InputTypeUtils
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.zip.GZIPOutputStream

val SESSION_REPLAY_ENABLED = SettingsKey(booleanPreferencesKey("sessionReplayEnabled"), false)

/**
 * Session replay for the keyboard: writes one gzipped JSONL file per input session to
 * getExternalFilesDir("replay"). Nothing here touches the network; files only leave the device
 * through `adb pull`. Off by default. See replay/README.md for the event format.
 *
 * Every method is called from the UI thread, and only captures values there; serialization and
 * file IO happen on a single worker so the keyboard never waits for the disk.
 */
object SessionRecorder {
    private const val FORMAT_VERSION = 1
    private const val TEXT_WINDOW = 256
    private const val FLUSH_EVERY = 256
    private const val RETENTION_DAYS = 90
    private const val MAX_TOTAL_BYTES = 1024L * 1024L * 1024L

    /** Banking, payment, authenticator and password apps are never recorded. */
    private val BLOCKED_PACKAGES = setOf(
        "com.agilebits.onepassword",
        "com.bitwarden.mobile", "com.x8bit.bitwarden",
        "com.google.android.apps.authenticator2",
        "com.authy.authy",
        "com.paypal.android.p2pmobile",
        "de.dkb.portalapp", "com.dkbcodefactory.banking",
        "de.fyrst.banking", "com.db.pwcc.dbmobile", "de.postbank.finanzassistent",
        "com.starfinanz.smob.android.sfinanzstatus", "de.fiducia.smartphone.android.banking.vr",
        "de.number26.android", "de.commerzbanking.mobil", "com.ing.diba.mbbr2",
        "com.android.systemui", "com.android.keyguard",
    )
    private val BLOCKED_SUBSTRINGS = listOf("bank", "banking", "password", "passwort", "authenticator", "wallet")

    @Volatile private var active: Session? = null
    private val worker = Executors.newSingleThreadExecutor { r ->
        Thread(r, "SessionRecorder").apply { isDaemon = true }
    }

    private class Session(val writer: BufferedWriter, val file: File) {
        var events = 0
        var keys = 0
        var lastLayout = 0
    }

    @JvmStatic
    fun isRecording(): Boolean = active != null

    /** Why this field may not be recorded, or null if it may. Never logs the field's content. */
    private fun refusalReason(info: EditorInfo): String? {
        val inputType = info.inputType
        if (InputTypeUtils.isPasswordInputType(inputType) ||
            InputTypeUtils.isVisiblePasswordInputType(inputType)) return "password"
        if ((inputType and InputType.TYPE_MASK_CLASS) != InputType.TYPE_CLASS_TEXT) return "nontext"
        if ((info.imeOptions and EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING) != 0) return "incognito"
        val attrs = Settings.getInstance().current?.mInputAttributes
        if (attrs != null && (attrs.mNoLearning || attrs.mIsPasswordField)) return "nolearning"
        val pkg = (info.packageName ?: "").lowercase(Locale.ROOT)
        if (pkg in BLOCKED_PACKAGES || BLOCKED_SUBSTRINGS.any { it in pkg }) return "blocked_app"
        return null
    }

    @JvmStatic
    fun startSession(context: Context, info: EditorInfo?, restarting: Boolean, locale: String?) {
        endSession("restart")
        if (info == null || !context.isDirectBootUnlocked) return
        if (!context.getSettingBlocking(SESSION_REPLAY_ENABLED)) return
        if (refusalReason(info) != null) return
        val dir = context.getExternalFilesDir("replay") ?: return

        val now = System.currentTimeMillis()
        val config = context.resources.configuration
        val metrics = context.resources.displayMetrics
        val appContext = context.applicationContext
        val head = JSONObject().apply {
            put("e", "start"); put("t", SystemClock.uptimeMillis()); put("v", FORMAT_VERSION)
            put("wall", now); put("build", BuildConfig.VERSION_NAME)
            put("pkg", info.packageName); put("inputType", info.inputType)
            put("imeOptions", info.imeOptions); put("field", info.fieldId)
            put("hint", info.hintText?.toString()); put("label", info.label?.toString())
            put("restarting", restarting); put("locale", locale)
            put("w", metrics.widthPixels); put("h", metrics.heightPixels)
            put("dpi", metrics.densityDpi)
            put("land", config.orientation == Configuration.ORIENTATION_LANDSCAPE)
        }
        val stamp = SimpleDateFormat("yyyy-MM-dd'T'HH-mm-ss-SSS", Locale.ROOT).format(Date(now))
        val file = File(dir, "$stamp.jsonl.gz")
        try {
            val writer = BufferedWriter(
                OutputStreamWriter(GZIPOutputStream(FileOutputStream(file), 8192, true), Charsets.UTF_8))
            active = Session(writer, file)
        } catch (e: Exception) {
            return
        }
        write(head)
        worker.execute { prune(appContext) }
    }

    @JvmStatic
    fun endSession(reason: String) {
        val s = active ?: return
        active = null
        val tail = JSONObject().apply {
            put("e", "end"); put("t", SystemClock.uptimeMillis()); put("reason", reason)
        }
        worker.execute {
            try {
                s.writer.write(tail.toString()); s.writer.newLine()
                s.writer.close()
                if (s.keys == 0) s.file.delete()   // the field was shown but nothing was typed
            } catch (_: Exception) {
            }
        }
    }

    private fun write(json: JSONObject) {
        val s = active ?: return
        worker.execute {
            try {
                s.writer.write(json.toString()); s.writer.newLine()
                if (++s.events % FLUSH_EVERY == 0) s.writer.flush()
            } catch (_: Exception) {
            }
        }
    }

    private inline fun event(name: String, fill: JSONObject.() -> Unit) {
        if (active == null) return
        write(JSONObject().apply { put("e", name); put("t", SystemClock.uptimeMillis()); fill() })
    }

    /** Raw touch stream, including batched history, in the view's own coordinates. */
    @JvmStatic
    fun touch(view: View, me: MotionEvent, source: String) {
        if (active == null) return
        val actionIndex = me.actionIndex
        val pointers = JSONArray()
        for (p in 0 until me.pointerCount) {
            val samples = JSONArray()
            for (h in 0 until me.historySize) {
                samples.put(JSONArray().put(me.getHistoricalEventTime(h))
                    .put(round(me.getHistoricalX(p, h))).put(round(me.getHistoricalY(p, h)))
                    .put(round(me.getHistoricalPressure(p, h))))
            }
            pointers.put(JSONObject().apply {
                put("id", me.getPointerId(p)); put("x", round(me.getX(p))); put("y", round(me.getY(p)))
                put("p", round(me.getPressure(p))); put("s", round(me.getSize(p)))
                put("maj", round(me.getTouchMajor(p))); put("min", round(me.getTouchMinor(p)))
                if (samples.length() > 0) put("hist", samples)
            })
        }
        val loc = IntArray(2)
        view.getLocationOnScreen(loc)
        val action = me.actionMasked
        val eventTime = me.eventTime
        val downTime = me.downTime
        event("touch") {
            put("src", source); put("a", action); put("i", actionIndex)
            put("et", eventTime); put("dt", downTime); put("ox", loc[0]); put("oy", loc[1])
            put("ps", pointers)
        }
    }

    /** Key rectangles, so every touch can be replayed against the exact layout that was visible. */
    @JvmStatic
    fun layout(view: View, keyboard: Keyboard, pendingFromView: Boolean) {
        val s = active ?: return
        val keys = JSONArray()
        for (key in keyboard.getSortedKeys()) {
            keys.put(JSONArray().put(key.code).put(key.label ?: JSONObject.NULL)
                .put(key.x).put(key.y).put(key.width).put(key.height))
        }
        val shift = TouchCorrection.shiftFor(keyboard)
        val signature = keys.toString().hashCode() * 31 + (shift?.contentHashCode() ?: 0)
        if (signature == s.lastLayout) return
        s.lastLayout = signature
        val id = keyboard.mId?.toString()
        val w = keyboard.mOccupiedWidth
        val h = keyboard.mOccupiedHeight
        val padLeft = view.paddingLeft
        val padTop = view.paddingTop
        event("layout") {
            put("id", id); put("kw", w); put("kh", h); put("pl", padLeft); put("pt", padTop)
            put("vw", view.width); put("vh", view.height); put("late", pendingFromView)
            put("keys", keys)
            // Touch correction moves touches by this many pixels before key detection; key events
            // carry the moved coordinates, touch events the raw ones.
            if (shift != null) put("shift", JSONArray().put(round(shift[0])).put(round(shift[1])))
        }
    }

    /** One decoded key press with the text state right after the keyboard handled it. */
    @JvmStatic
    fun key(ev: Event, composing: String?, before: CharSequence?, selStart: Int, selEnd: Int) {
        val s = active ?: return
        s.keys++
        val code = ev.mCodePoint
        val keyCode = ev.mKeyCode
        val x = ev.mX
        val y = ev.mY
        val text = before?.toString()
        event("key") {
            put("cp", code); put("kc", keyCode); put("x", x); put("y", y)
            put("w", composing ?: JSONObject.NULL); put("txt", text ?: JSONObject.NULL)
            put("ss", selStart); put("se", selEnd)
        }
    }

    @JvmStatic
    fun suggestions(words: SuggestedWords, composing: String?) {
        if (active == null) return
        val list = JSONArray()
        for (i in 0 until words.size()) {
            val w = words.getInfo(i)
            list.put(JSONArray().put(w.mWord).put(w.mScore).put(w.kind)
                .put(w.mSourceDict?.mDictType ?: JSONObject.NULL))
        }
        val auto = words.mWillAutoCorrect
        val typedValid = words.mTypedWordValid
        var fromLm = 0
        for (i in 0 until words.size()) if (words.getInfo(i).mOriginatesFromTransformerLM) fromLm++
        event("sugg") {
            put("w", composing ?: JSONObject.NULL); put("auto", auto); put("valid", typedValid)
            put("lm", fromLm)
            put("l", list)
        }
    }

    @JvmStatic
    fun pick(word: String, index: Int, composing: String?) {
        event("pick") { put("w", composing ?: JSONObject.NULL); put("to", word); put("i", index) }
    }

    /** A finished word. `typed` differing from `word` is an autocorrection or a picked suggestion. */
    @JvmStatic
    fun commit(typed: String?, word: String, commitType: Int, separator: String?, before: CharSequence?) {
        val s = active ?: return
        s.keys++    // swipe sessions never produce key events
        val text = before?.toString()
        event("commit") {
            put("typed", typed ?: JSONObject.NULL); put("to", word); put("ct", commitType)
            put("sep", separator ?: JSONObject.NULL); put("txt", text ?: JSONObject.NULL)
        }
    }

    /** The user rejected the last autocorrection with backspace. The most valuable label we have. */
    @JvmStatic
    fun revert(typed: String?, word: String?) {
        event("revert") {
            put("typed", typed ?: JSONObject.NULL); put("to", word ?: JSONObject.NULL)
        }
    }

    @JvmStatic
    fun selection(oldStart: Int, oldEnd: Int, newStart: Int, newEnd: Int, candStart: Int, candEnd: Int) {
        event("sel") {
            put("os", oldStart); put("oe", oldEnd); put("ns", newStart); put("ne", newEnd)
            put("cs", candStart); put("ce", candEnd)
        }
    }

    @JvmStatic
    fun textWindow(): Int = TEXT_WINDOW

    private fun round(v: Float): Double = Math.round(v * 100f) / 100.0

    private fun files(context: Context): List<File> =
        context.getExternalFilesDir("replay")?.listFiles()?.filter { it.isFile }.orEmpty()

    private fun prune(context: Context) {
        val cutoff = System.currentTimeMillis() - RETENTION_DAYS * 86_400_000L
        val all = files(context).sortedBy { it.lastModified() }
        var total = all.sumOf { it.length() }
        for (f in all) {
            if (f.lastModified() < cutoff || total > MAX_TOTAL_BYTES) {
                total -= f.length()
                f.delete()
            }
        }
    }

    /** Stops the running session and removes everything recorded so far. */
    @JvmStatic
    fun deleteAll(context: Context): Int {
        endSession("delete")
        val appContext = context.applicationContext
        return try {
            worker.submit<Int> { files(appContext).count { it.delete() } }.get()
        } catch (e: Exception) {
            0
        }
    }

    fun stats(context: Context): Pair<Int, Long> {
        val all = files(context)
        return all.size to all.sumOf { it.length() }
    }
}
