package org.deeplearning4j.word2vec;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.Collection;
import java.util.List;

/**
 * sake2vec@GUI版
 * Created by b1012059 on 2015/04/25.
 * @author b1012059 Wataru Matsudate
 */
public class ApplicationMain{

    /**
     * Mainクラス
     * @param args
     */
    public static void main(String[] args) {
        OpenFrame frm = new OpenFrame("Test sake2vec");
        frm.setLocation(300, 200);
        frm.setSize(600, 400);
        frm.setBackground(Color.LIGHT_GRAY);
        frm.setVisible(true);
    }
}

class OpenFrame extends Frame implements ActionListener {

    MenuItem mil1, mil2;
    //Label lb1;
    TextArea txtar1;
    TextField tf1, tf2;
    Sake2Vec sake2Vec;
    String word1, word2;
    Panel panel1, panel2;
    Button button1;

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
        add(panel2, BorderLayout.SOUTH);
        panel2.setLayout((new FlowLayout()));

        //textエリア関連
        txtar1 = new TextArea(40,50);
        txtar1.setFont(new Font("Dialog", Font.PLAIN, 18));
        txtar1.setForeground(new Color(64, 64, 64));
        panel1.add(txtar1);

        //textフィールド関連
        tf1 = new TextField("", 20);
        tf2 = new TextField("", 20);
        panel2.add(tf1);
        panel2.add(tf2);

        //button関連
        button1 = new Button("実行");
        button1.addActionListener(this);
        panel2.add(button1);

    }

    /**
     * menu等のボタンAction処理
     * @param e Actionイベントの発生
     */
    public void actionPerformed(ActionEvent e) {
        Object obj = e.getSource();

        //menu処理
        if (obj == mil1) {
            /*OpenFrame frm = new OpenFrame("Test sake2vec");
            frm.setLocation(300, 200);
            frm.setSize(600, 400);
            frm.setBackground(Color.LIGHT_GRAY);
            frm.setVisible(true);*/
            System.out.println("デバッグ用");
        } else if (obj == mil2) {
            try {
                WindowTest();
            } catch (Exception e1) {
                System.out.println("エラーが発生しました。");
                e1.printStackTrace();
            }
        } else if(obj == button1){
            try {
                word1 = "people";
                word2 = "money";
                Sake2vecRun(word1, word2);
            } catch (Exception e1) {
                System.out.println("エラーが発生しました。ファイルを参照していない可能性があります");
                e1.printStackTrace();
            }
        }
    }

    /**
     * fileDialogの生成及びsake2vecの実行メソッド
     * @throws Exception
     */
    private void WindowTest() throws Exception {
        FileDialog fileDialog = new FileDialog(this);
        fileDialog.setVisible(true);
        //String dir = fileDialog.getDirectory();
        String fileName = fileDialog.getFile();

        //fileDialogによってtxtデータが参照されたら
        if (fileName != null) {
            //sake2vecのインスタンスを作成
            sake2Vec = new Sake2Vec(fileName);
        }
    }

    private void Sake2vecRun(String setWord1, String setWord2) throws Exception {
        int number = 20;

        if(true) {
            //sake2vec本体の実行
            sake2Vec.Sake2vecExample();
            //setWord1とsetWord2の近似値
            txtar1.append(setWord1 + "と" + setWord2 + "の類似値は" +
                    String.valueOf(sake2Vec.sake2vecSimilarity(setWord1, setWord2)) + "\n");

            /*//setWord1に近い単語をnumber個一括表示
            List<String> tmp = (List<String>) sake2Vec.sake2vecWordsNearest(number);
            txtar1.append(setWord1 + "に近い" + number + "個の単語は" + tmp + "\n");

            //setWord1と上記各単語の近似値
            txtar1.append("各単語の類似値を表示します" + "\n");
            double[] tmpData = sake2Vec.sake2vecWordsNearestCustom(number);
            for (int i = 0; i < number; i++) {
                txtar1.append(setWord1 + "と" + tmp.get(i) + "の類似値は" + tmpData[i] + "\n");
            }*/
            txtar1.append("\n");
        }
    }

        /*/ファイル読み込み用デバッグ部分
       try{
            String s;
            int i = 1;
            FileReader rd = new FileReader(dir + fileName);
            BufferedReader br = new BufferedReader(rd);

            txtar1.append("ファイル読み込みデバッグ用\n");
            while((s = br.readLine()) != null){
                if(i != 100) {
                    txtar1.append(s + "\n");
                    i++;
                } else {
                    break;
                }
            }

           //ファイルクロージング処理
            br.close();
            rd.close();

        }catch(IOException e){
            //エラーが発生したら　エラーを表示
            System.out.println("Err=" + e);
        }*/
}
