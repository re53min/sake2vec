package org.deeplearning4j.word2vec;

import java.awt.*;
import java.awt.event.*;
import java.io.*;

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
    Sake2Vec sake2Vec;
    String word1, word2;

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

        //textエリア関連
        txtar1 = new TextArea();
        txtar1.setFont(new Font("Dialog", Font.PLAIN, 18));
        txtar1.setForeground(new Color(64, 64, 64));
        add(txtar1, BorderLayout.CENTER);

        //panel関連
        Panel pn1 = new Panel();
        pn1.setLayout(new GridLayout(1, 3));

        add(pn1, BorderLayout.SOUTH);
    }

    /**
     * menu等のボタンAction処理
     * @param e Actionイベントの発生
     */
    public void actionPerformed(ActionEvent e) {
        Object obj = e.getSource();

        //menu処理
        if (obj == mil1) {
            System.out.println("デバッグ用");
        } else if (obj == mil2) {
            word1 = "people";
            word2 = "money";
            try {
                WindowTest(word1, word2);
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        }
    }

    /**
     * fileDialogの生成及びsake2vecの実行メソッド
     * @param setWord1
     * @param setWord2
     * @throws Exception
     */
    private void WindowTest(String setWord1, String setWord2) throws Exception {
        FileDialog fileDialog = new FileDialog(this);
        fileDialog.setVisible(true);
        String dir = fileDialog.getDirectory();
        String fileName = fileDialog.getFile();
        int number = 20;

        //fileDialogによってtextデータが参照されたら
        if (fileName != null) {
            //sake2vecのインスタンスを作成
            sake2Vec = new Sake2Vec(fileName, setWord1, setWord2);
            //sake2vec本体の実行
            sake2Vec.Sake2vecExample();
            //setWord1とsetWord2の近似値表示
            txtar1.append(setWord1 + "と" + setWord2 + "の近似値は" +
                    String.valueOf(sake2Vec.sake2vecSimilarity()) + "\n");
            //setWord1に近い単語20個表示
            txtar1.append(sake2Vec.sake2vecWordsNearest(number) + "\n");
            txtar1.append("\n");
        }

        //ファイル読み込み用デバッグ部分
       try{
            String s;
            int i = 1;
            FileReader rd = new FileReader(dir + fileName);
            BufferedReader br = new BufferedReader(rd);

            /*txtar1.append("ファイル読み込みデバッグ用\n");
            while((s = br.readLine()) != null){
                if(i != 100) {
                    txtar1.append(s + "\n");
                    i++;
                } else {
                    break;
                }
            }*/

           //ファイルクロージング処理
            br.close();
            rd.close();

        }catch(IOException e){
            //エラーが発生したら　エラーを表示
            System.out.println("Err=" + e);
        }
    }
}
