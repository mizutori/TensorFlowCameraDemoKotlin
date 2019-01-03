package com.goldrushcomputing.tensorflowcamerademokotlin

import android.os.Bundle
import android.support.v7.app.AppCompatActivity

class CamaraActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camara)

        if (null == savedInstanceState) {
            supportFragmentManager
                .beginTransaction()
                .replace(R.id.container, CameraFragment.newInstance())
                .commit()
        }
    }
}
