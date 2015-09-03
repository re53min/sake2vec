package org.deeplearning4j.word2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.List;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.*;

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
    private static Logger log = LoggerFactory.getLogger(Sake2Vec.class);

    private MenuItem mil1, mil2;
    private TextArea txtar1;
    private JTextField tf1, tf2, tf3;
    private Panel panel1, panel2;
    private Button button1;

    private Sake2Vec2 vec;
    private ChangerInput input;

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
        mil1 = new MenuItem("新規");
        mil2 = new MenuItem("開く");
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
            /*OpenFrame frm = new OpenFrame("Test sake2vec");
            frm.setLocation(300, 200);
            frm.setSize(600, 400);
            frm.setBackground(Color.LIGHT_GRAY);
            frm.setVisible(true);*/
            System.out.println("デバッグ用");
        } else if (obj == mil2) {
            try {
                SendFileName();
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        } else if(obj == button1) {
            try {
                sentence = tf1.getText();
                tf1.setText("");
                ChangerInputRun(sentence);
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
    private void SendFileName() throws Exception {
        FileDialog fileDialog = new FileDialog(this);
        fileDialog.setVisible(true);
        String fileName = fileDialog.getFile();

        //fileDialogによってfileが参照されたら
        if (fileName != null) {
            input = new ChangerInput(fileName);
        }
    }

    private void ChangerInputRun(String sentence) throws Exception {
        input = new ChangerInput();
        txtar1.append(input.transRun(sentence));
        txtar1.append(input.output());
        txtar1.append("\n");
    }
    /*private void Sake2vecRun(String posiWord, String negaWord, int number, int flag) throws Exception {
        java.util.List<String> posi = new ArrayList();
        java.util.List<String> nega = new ArrayList();

        //sake2vec本体の実行
        vec.Sake2vecExample();

        //テキストフィールド内(positive, negative)の意味演算
        String[] posiTmp = posiWord.split(",", -1);
        for(int i = 0; i < posiTmp.length; i++){
            posi.add(posiTmp[i]);
            log.info("positive words:" + posi.get(i));
        }
        String[] negaTmp = negaWord.split(",", -1);
        for(int i = 0; i < negaTmp.length; i++){
            nega.add(negaTmp[i]);
            log.info("negative words:" + nega.get(i));
        }

        if (flag == 1) {
            //similarity of word1 and word2
            double simResult = vec.sake2vecSimilarity(posi.get(0), nega.get(0));
            log.info("Similarity between " + posi.get(0) + " and " + nega.get(0) + ": " + simResult);
            txtar1.append("Similarity between " + posi.get(0) + " and " + nega.get(0) + ": " + simResult+ "\n");
        } else if (flag == 2 ) {
            //個数表示
            java.util.List<String> nearResult = (java.util.List<String>) vec.sake2vecWordsNearest(posi.get(0), number);
            log.info("Word Nearest " + posi.get(0) + " is: " + nearResult);
            //結果の表示
            txtar1.append("演算結果: " + nearResult + "\n");
        } else if (flag == 3){
            //意味演算実行
            java.util.List<String> nearResult = (java.util.List<String>) vec.sake2vecWordsNearest(posi, nega, number);
            log.info("Word Nearest: " + nearResult);
            //結果の表示
            txtar1.append("演算結果: " + nearResult + "\n");
        }

        txtar1.append("\n");
    }*/

}
