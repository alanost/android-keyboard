package org.futo.inputmethod.latin.replay

import android.content.Context
import android.view.MotionEvent
import androidx.datastore.preferences.core.booleanPreferencesKey
import org.futo.inputmethod.keyboard.Keyboard
import org.futo.inputmethod.latin.uix.SettingsKey
import org.futo.inputmethod.latin.uix.getSettingBlocking
import org.futo.inputmethod.latin.uix.isDirectBootUnlocked
import org.json.JSONObject
import java.io.File

val TOUCH_CORRECTION_ENABLED = SettingsKey(booleanPreferencesKey("touchCorrectionEnabled"), false)

/**
 * Personal touch offset correction. Session-replay data shows how far a user's touches land from
 * the key they aim at. When enabled, every touch on the main keyboard is moved back by that offset
 * before key detection, so the intended key is hit more often. Off by default.
 *
 * The offset is read from getExternalFilesDir(null)/calibration.json, written via adb:
 *     {"dx": 0.0, "dy": 0.1}
 * in units of the key pitch (most common key width/height including the gap), positive meaning the
 * user lands right of / below the key centre. Without the file nothing changes.
 *
 * Session replay keeps recording the raw touch stream; the applied shift is stored in each layout event.
 */
object TouchCorrection {
    private const val LIMIT = 0.5f

    @Volatile private var enabled = false
    @Volatile private var dxPitch = 0f
    @Volatile private var dyPitch = 0f
    @Volatile private var fileStamp = Long.MIN_VALUE

    /** Called when an input field starts; picks up the setting and a changed calibration file. */
    @JvmStatic
    fun reload(context: Context) {
        // Before the first unlock neither the settings nor external storage can be read.
        if (!context.isDirectBootUnlocked) {
            enabled = false
            return
        }
        enabled = context.getSettingBlocking(TOUCH_CORRECTION_ENABLED)
        val dir = context.getExternalFilesDir(null) ?: return
        val file = File(dir, "calibration.json")
        val stamp = if (file.isFile) file.lastModified() else 0L
        if (stamp == fileStamp) return
        fileStamp = stamp
        var dx = 0f
        var dy = 0f
        if (stamp != 0L) {
            try {
                val json = JSONObject(file.readText())
                dx = json.optDouble("dx", 0.0).toFloat()
                dy = json.optDouble("dy", 0.0).toFloat()
                if (dx.isNaN() || dy.isNaN()) { dx = 0f; dy = 0f }
            } catch (_: Exception) {
                dx = 0f; dy = 0f
            }
        }
        dxPitch = dx.coerceIn(-LIMIT, LIMIT)
        dyPitch = dy.coerceIn(-LIMIT, LIMIT)
    }

    /** Shift in pixels that is added to every touch on [keyboard], or null when nothing is corrected. */
    @JvmStatic
    fun shiftFor(keyboard: Keyboard): FloatArray? {
        if (!enabled || (dxPitch == 0f && dyPitch == 0f)) return null
        return floatArrayOf(-dxPitch * keyboard.mMostCommonKeyWidth, -dyPitch * keyboard.mMostCommonKeyHeight)
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
