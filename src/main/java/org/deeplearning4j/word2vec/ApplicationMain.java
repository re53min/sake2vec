package org.deeplearning4j.word2vec;

/**
 * Created by b1012059 on 2015/04/25.
 */

import java.awt.*;
import java.awt.event.*;
import java.io.*;

public class ApplicationMain{
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
    Sake2vec sake2vec;
    String word1, word2;

    public OpenFrame(String title) {
        setTitle(title);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                System.exit(0);
            }
        });

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

        txtar1 = new TextArea();
        txtar1.setFont(new Font("Dialog", Font.PLAIN, 18));
        txtar1.setForeground(new Color(64, 64, 64));
        add(txtar1, BorderLayout.CENTER);

        Panel pn1 = new Panel();
        pn1.setLayout(new GridLayout(1, 3));

        add(pn1, BorderLayout.SOUTH);
    }

    public void actionPerformed(ActionEvent e) {
        Object obj = e.getSource();

        if (obj == mil1) {
            System.out.println("デバッグ用");
        } else if (obj == mil2) {
            word1 = "people";
            word2 = "money";
            WindowTest(word1, word2);
        }
    }

    private void WindowTest(String setWord1, String setWord2){
        FileDialog fileDialog = new FileDialog(this);
        fileDialog.setVisible(true);
        String dir = fileDialog.getDirectory();
        String fileName = fileDialog.getFile();

        if(fileName == null) {
        } else {
            sake2vec = new Sake2vec(fileName, setWord1, setWord2);
            txtar1.append(sake2vec.sake2vecResult() + "\n");
            txtar1.append("\n");
        }

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

            br.close();
            rd.close();

        }catch(IOException e){
            //エラーが発生したら　エラーを表示
            System.out.println("Err=" + e);
        }
    }
}
