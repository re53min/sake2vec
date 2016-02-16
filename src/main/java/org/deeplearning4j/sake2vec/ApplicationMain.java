package org.deeplearning4j.sake2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * sake2vec@GUI版_v2
 * Created by b1012059 on 2015/08/11.
 * @author Wataru Matsudate
 */

public class ApplicationMain {


    /**
     * Mainクラス
     * @param args
     */
    public static void main(String[] args) {
        OpenFrame frm = new OpenFrame("Sake2Vec(仮)");
        frm.setLocation(300, 200);
        frm.setSize(650, 400);
        frm.setBackground(Color.LIGHT_GRAY);
        frm.setVisible(true);
    }
}

class OpenFrame extends Frame implements ActionListener {
    private static Logger log = LoggerFactory.getLogger(ApplicationMain.class);

    private MenuItem mil1, mil2;
    private TextArea txtar1;
    private JTextField tf1;
    private Panel panel1, panel2;
    private Button button1;

    /**
     * OpenFrameメソッド
     * GUI等の実装部
     * @param title フレームタイトル
     */
    public OpenFrame(String title) {
        setTitle(title);

        //windowのクロージング処理
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                System.exit(0);
            }
        });

        //menu関連
        MenuBar mb = new MenuBar();
        Menu mn1 = new Menu("ファイル");
        mil1 = new MenuItem("モデル");
        mil2 = new MenuItem("コーパス");
        mil1.addActionListener(this);
        mil2.addActionListener(this);
        mn1.add(mil1);
        mn1.add(mil2);
        mb.add(mn1);
        setMenuBar(mb);

        //panel関連
        panel1 = new Panel();
        panel2 = new Panel();
        add(panel1);
        add(panel2);

        //panelレイアウト
        setLayout(new FlowLayout());

        //textフィールド関連
        tf1 = new JTextField(50);
        panel1.add(tf1);

        //button関連
        button1 = new Button("送信");
        button1.addActionListener(this);
        panel1.add(button1);

        //textエリア関連
        txtar1 = new TextArea();
        txtar1.setFont(new Font("Dialog", Font.PLAIN, 18));
        //txtar1.setForeground(new Color(64, 64, 64));
        panel2.add("Center", txtar1);

    }


    /**
     * menu等のGUIAction処理
     * @param e Actionイベントの発生
     */
    public void actionPerformed(ActionEvent e) {
        String fileName = null;
        boolean flag = true;
        Object obj = e.getSource();

        //action処理
        if (obj == mil1) {
            try {
                fileName = sendFileName();
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        } else if (obj == mil2) {
            try {
                flag = false;
                fileName = sendFileName();
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        } else if(obj == button1) {
            try {
                String query = tf1.getText();
                tf1.setText("");
                log.info("入力:" + query);
                Sake2Vec sake2vec = new Sake2Vec(query, fileName, flag);
                sakeChangerRun(sake2vec);
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        }
    }

    /**
     * model-true, corpus-false
     * @return
     * @throws Exception
     */
    private String sendFileName() {
        FileDialog fileDialog = new FileDialog(this);
        fileDialog.setVisible(true);
        String fileName = fileDialog.getFile();

        log.info("ファイルを参照");

        return fileName;
    }


    /**
     *
     * @param sake2vec
     * @throws Exception
     */
    private void sakeChangerRun(Sake2Vec sake2vec) throws Exception {

        txtar1.append("あなた： " + sake2vec.getQuery() + "\n");
        txtar1.append("システム： " + sake2vec.getResult());
        txtar1.append("\n");
    }
}
