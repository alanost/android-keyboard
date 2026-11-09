package org.futo.inputmethod.latin.replay

import android.content.Context
import android.view.MotionEvent
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import org.futo.inputmethod.keyboard.Keyboard
import org.futo.inputmethod.latin.uix.SettingsKey
import org.futo.inputmethod.latin.uix.getSettingBlocking
import org.futo.inputmethod.latin.uix.isDirectBootUnlocked
import org.futo.inputmethod.latin.uix.setSettingBlocking
import org.json.JSONObject
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt

val TOUCH_CORRECTION_ENABLED = SettingsKey(booleanPreferencesKey("touchCorrectionEnabled"), false)

/** How far the user lands from the key centre, in percent of the key size; positive = right of / below. */
val TOUCH_CORRECTION_DX = SettingsKey(intPreferencesKey("touchCorrectionDx"), 0)
val TOUCH_CORRECTION_DY = SettingsKey(intPreferencesKey("touchCorrectionDy"), 0)

/**
 * Personal touch offset correction. Session-replay data shows how far a user's touches land from
 * the key they aim at. When enabled, every touch on the main keyboard is moved back by that offset
 * before key detection, so the intended key is hit more often. Off by default.
 *
 * The offsets are set with two sliders in the prediction settings, in percent of the most common key
 * width/height (Keyboard.mMostCommonKeyWidth/Height, the visible key size). Measured values can also
 * be pushed via adb as getExternalFilesDir(null)/calibration.json, e.g.
 * {"dx": 0, "dy": 10, "enabled": true}; the file is imported into the settings when the next input
 * field starts and then deleted.
 *
 * Session replay keeps recording the raw touch stream; the applied shift is stored in each layout event.
 */
object TouchCorrection {
    /** Largest offset in either direction, in percent of the key size. */
    const val LIMIT_PERCENT = 30
    private const val FILE_NAME = "calibration.json"

    @Volatile private var enabled = false
    @Volatile private var dxPercent = 0
    @Volatile private var dyPercent = 0

    /** Key size of the last keyboard that was touched, so the settings can show pixels (0 = unknown yet). */
    @Volatile private var lastKeyWidth = 0
    @Volatile private var lastKeyHeight = 0

    /** Called when an input field starts; picks up the settings and imports a pushed calibration file. */
    @JvmStatic
    fun reload(context: Context) {
        // Before the first unlock neither the settings nor external storage can be read.
        if (!context.isDirectBootUnlocked) {
            enabled = false
            return
        }
        var on = context.getSettingBlocking(TOUCH_CORRECTION_ENABLED)
        var dx = context.getSettingBlocking(TOUCH_CORRECTION_DX)
        var dy = context.getSettingBlocking(TOUCH_CORRECTION_DY)

        val file = context.getExternalFilesDir(null)?.let { File(it, FILE_NAME) }
        if (file != null && file.isFile) {
            // The settings cache updates asynchronously, so this session uses the imported values directly.
            try {
                val json = JSONObject(file.readText())
                if (json.has("dx")) {
                    dx = clampPercent(json.getDouble("dx"))
                    context.setSettingBlocking(TOUCH_CORRECTION_DX.key, dx)
                }
                if (json.has("dy")) {
                    dy = clampPercent(json.getDouble("dy"))
                    context.setSettingBlocking(TOUCH_CORRECTION_DY.key, dy)
                }
                if (json.has("enabled")) {
                    on = json.getBoolean("enabled")
                    context.setSettingBlocking(TOUCH_CORRECTION_ENABLED.key, on)
                }
            } catch (_: Exception) {
                // Unreadable file: the settings stay as they are.
            }
            file.delete()   // consumed: from now on the sliders own the values
        }

        enabled = on
        dxPercent = dx.coerceIn(-LIMIT_PERCENT, LIMIT_PERCENT)
        dyPercent = dy.coerceIn(-LIMIT_PERCENT, LIMIT_PERCENT)
    }

    private fun clampPercent(value: Double): Int =
        if (value.isNaN()) 0 else value.roundToInt().coerceIn(-LIMIT_PERCENT, LIMIT_PERCENT)

    /** Shift in pixels that is added to every touch on [keyboard], or null when nothing is corrected. */
    @JvmStatic
    fun shiftFor(keyboard: Keyboard): FloatArray? {
        lastKeyWidth = keyboard.mMostCommonKeyWidth
        lastKeyHeight = keyboard.mMostCommonKeyHeight
        if (!enabled || (dxPercent == 0 && dyPercent == 0)) return null
        return floatArrayOf(
            -dxPercent / 100f * keyboard.mMostCommonKeyWidth,
            -dyPercent / 100f * keyboard.mMostCommonKeyHeight
        )
    }

    /** Slider label: "+12 % ≈ 19 px" once a keyboard has been used, otherwise "+12 %". */
    fun describe(percent: Int, vertical: Boolean): String {
        val sign = if (percent > 0) "+" else ""
        val size = if (vertical) lastKeyHeight else lastKeyWidth
        val pixels = if (size > 0 && percent != 0) " ≈ ${(abs(percent) * size / 100f).roundToInt()} px" else ""
        return "$sign$percent %$pixels"
    }

    /**
     * Returns a shifted copy of [event] that the caller must recycle, or null when no correction
     * applies. The shift is reduced where it would push a pointer that is on the keyboard off it,
     * so taps along the edges are never lost.
     */
    @JvmStatic
    fun correct(event: MotionEvent, keyboard: Keyboard, width: Int, height: Int): MotionEvent? {
        val shift = shiftFor(keyboard) ?: return null
        var dx = shift[0]
        var dy = shift[1]
        val maxX = (width - 1).toFloat()
        val maxY = (height - 1).toFloat()
        for (i in 0 until event.pointerCount) {
            val x = event.getX(i)
            val y = event.getY(i)
            if (x in 0f..maxX) dx = dx.coerceIn(-x, maxX - x)
            if (y in 0f..maxY) dy = dy.coerceIn(-y, maxY - y)
        }
        if (dx == 0f && dy == 0f) return null
        val copy = MotionEvent.obtain(event)
        copy.offsetLocation(dx, dy)
        return copy
    }
}
