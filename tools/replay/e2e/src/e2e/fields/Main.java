package e2e.fields;

import android.app.Activity;
import android.os.Bundle;
import android.text.InputType;
import android.widget.EditText;

/** One text field of the kind named by the "field" extra, so the test can drive the keyboard. */
public class Main extends Activity {
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        String field = getIntent().getStringExtra("field");
        EditText edit = new EditText(this);
        if ("password".equals(field)) {
            edit.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        } else if ("number".equals(field)) {
            edit.setInputType(InputType.TYPE_CLASS_NUMBER);
        } else {
            edit.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        }
        setContentView(edit);
        edit.requestFocus();
    }
}
