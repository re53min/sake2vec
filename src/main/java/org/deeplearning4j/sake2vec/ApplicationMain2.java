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

public class ApplicationMain2 {


    /**
     * Mainクラス
     * @param args
     */
    public static void main(String[] args) {
        OpenFrame2 frm = new OpenFrame2("Sake2Vec(仮)");
        frm.setLocation(300, 200);
        frm.setSize(650, 400);
        frm.setBackground(Color.LIGHT_GRAY);
        frm.setVisible(true);
    }
}

class OpenFrame2 extends Frame implements ActionListener {
    private static Logger log = LoggerFactory.getLogger(ApplicationMain2.class);

    private MenuItem mil1, mil2;
    private TextArea txtar1;
    private JTextField tf1, tf2, tf3;
    private Panel panel1, panel2;
    private Button button1;

    private Sake2Vec2 vec;
    private SakeChanger input;

    /**
     * OpenFrameメソッド
     * GUI等の実装部
     * @param title フレームタイトル
     */
    public OpenFrame2(String title) {
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
        mil1 = new MenuItem("コーパス");
        mil2 = new MenuItem("モデル");
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
        String sentence;
        int flag  = 0;
        int number = 0;
        Object obj = e.getSource();

        //action処理
        if (obj == mil1) {
            try {
                sendFileName();
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        } else if (obj == mil2) {
            try {
                sendModelName();
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        } else if(obj == button1) {
            try {
                sentence = tf1.getText();
                tf1.setText("");
                log.info("入力:" + sentence);
                sakeChangerRun(sentence);
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        }
    }


    /**
     * fileDialogの生成及びsake2vecのインスタンス生成
     * fileDialogで受け取ったfilenameをsake2vecに渡す
     * @throws Exception
     */
    private void sendFileName() throws Exception {
        FileDialog fileDialog = new FileDialog(this);
        fileDialog.setVisible(true);
        String fileName = fileDialog.getFile();

        //fileDialogによってcorpus fileが参照されたら
        if (fileName != null) {
            input = new SakeChanger(fileName, false);
            log.info("コーパスデータを参照");
        }
    }

    /**
     * fileDialogの生成及びsake2vecのインスタンス生成
     * fileDialogで受け取ったmodelnameをsake2vecに渡す
     * @throws Exception
     */
    private void sendModelName() throws Exception{
        FileDialog fileDialog = new FileDialog(this);
        fileDialog.setVisible(true);
        String modelName = fileDialog.getFile();

        //fileDialogによってmodel fileが参照されたら
        if (modelName != null) {
            input = new SakeChanger(modelName, true);
            log.info("モデルデータを参照");
        }
    }

    /**
     *
     * @param sentence
     * @throws Exception
     */
    private void sakeChangerRun(String sentence) throws Exception {
        //if(input == null) input = new ChangerSentence();

        txtar1.append(input.input(sentence)+ "\n");
        txtar1.append(input.output());
        txtar1.append("\n");
    }
}
