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
        txtar1.setFont(new Font("Dialog", Font.PLAIN, 14));
        txtar1.setForeground(new Color(64, 64, 64));
        add(txtar1, BorderLayout.CENTER);

        Panel pn1 = new Panel();
        pn1.setLayout(new GridLayout(1, 3));

        add(pn1, BorderLayout.SOUTH);
    }

    public void actionPerformed(ActionEvent e) {
        Object obj = e.getSource();

        if (obj == mil1) {
            System.out.println("とりあえず");
        } else if (obj == mil2) {
            WindowTest();
        }
    }

    private void WindowTest(){
        FileDialog fileDialog = new FileDialog(this);
        fileDialog.setVisible(true);
        String dir = fileDialog.getDirectory();
        String fileName = fileDialog.getFile();

        if(fileName == null) {
        } else {
            sake2vec = new Sake2vec(fileName);
            String result = sake2vec.sake2vecResult();
            txtar1.append(result + "\n");
        }

        try{
            String s;
            int i = 1;
            FileReader rd = new FileReader(dir + fileName);
            BufferedReader br = new BufferedReader(rd);

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
