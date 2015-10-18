package org.deeplearning4j.word2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.List;


/**
 * sake2vec@GUI版
 * Created by b1012059 on 2015/04/25.
 * @author Wataru Matsudate
 */
public class ApplicationMain{

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
    private static Logger log = LoggerFactory.getLogger(Sake2Vec.class);

    private MenuItem mil1, mil2;
    private TextArea txtar1;
    private JTextField tf1, tf2, tf3;
    private Panel panel1, panel2, panel3;
    private Button button1;
    private JRadioButton radio1, radio2, radio3;
    private ButtonGroup group;

    private Sake2Vec vec;

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
        panel3 = new Panel();
        add(panel1);
        add(panel2);
        add(panel3);

        //panelレイアウト
        setLayout(new FlowLayout());

        //radioボタン関連
        radio1 = new JRadioButton("Similar");
        radio2 = new JRadioButton("Near");
        radio3 = new JRadioButton("Near(posi, nega)");
        panel1.add(radio1);
        panel1.add(radio2);
        panel1.add(radio3);
        radio1.addActionListener(this);
        radio2.addActionListener(this);
        radio3.addActionListener(this);


        //textフィールド関連
        tf1 = new JTextField(25);
        tf2 = new JTextField(25);
        tf3 = new JTextField(15);
        tf3.setText("0");
        panel2.add(tf1);
        panel2.add(tf2);
        panel1.add(tf3);

        //button関連
        button1 = new Button("実行");
        button1.addActionListener(this);
        panel2.add(button1);

        //textエリア関連
        txtar1 = new TextArea();
        txtar1.setFont(new Font("Dialog", Font.PLAIN, 18));
        //txtar1.setForeground(new Color(64, 64, 64));
        panel3.add("Center", txtar1);

    }


    /**
     * menu等のGUIAction処理
     * @param e Actionイベントの発生
     */
    public void actionPerformed(ActionEvent e) {
        String posiWord, negaWord;
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
        } else if(radio1.equals("Similar")){
            flag = 1;
        } else if(obj == button1) {
            try {
                posiWord = tf1.getText();
                negaWord = tf2.getText();
                number = Integer.parseInt(tf3.getText());
                Sake2vecRun(posiWord, negaWord, number, flag);
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
        //String dir = fileDialog.getDirectory();
        String fileName = fileDialog.getFile();

        //fileDialogによってfileが参照されたら
        if (fileName != null) {
            //sake2vecのインスタンスを作成
            vec = new Sake2Vec(fileName);
        }
    }

    /**
     * Sake2Vec実行メソッド
     * @param posiWord positive word
     * @param negaWord negative word
     * @param number   number
     * @param flag     flag
     * @throws Exception
     */
    private void Sake2vecRun(String posiWord, String negaWord, int number, int flag) throws Exception {
        List<String> posi = new ArrayList();
        List<String> nega = new ArrayList();

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
            double simResult = vec.sakeSimilar(posi.get(0), nega.get(0));
            log.info("Similarity between " + posi.get(0) + " and " + nega.get(0) + ": " + simResult);
            txtar1.append("Similarity between " + posi.get(0) + " and " + nega.get(0) + ": " + simResult+ "\n");
        } else if (flag == 2 ) {
            //個数表示
            List<String> nearResult = (List<String>) vec.sakeWordsNearest(posi.get(0), number);
            log.info("Word Nearest " + posi.get(0) + " is: " + nearResult);
            //結果の表示
            txtar1.append("演算結果: " + nearResult + "\n");
        } else if (flag == 3){
            //意味演算実行
            List<String> nearResult = (List<String>) vec.sakeWordsNearest(posi, nega, number);
            log.info("Word Nearest: " + nearResult);
            //結果の表示
            txtar1.append("演算結果: " + nearResult + "\n");
        }

        txtar1.append("\n");
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
