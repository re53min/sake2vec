package org.deeplearning4j.word2vec;

/**
 * Created by b1012059 on 2015/04/23.
 */

import javax.swing.*;
import java.io.File;
import java.awt.BorderLayout;
import java.awt.event.*;

public class JFileChooserTest extends JFrame implements ActionListener{

    JLabel label;

    public static void main(String[] args){
        JFileChooserTest frame = new JFileChooserTest();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setBounds(10, 10, 300, 200);
        frame.setTitle("タイトル");
        frame.setVisible(true);
    }

    JFileChooserTest(){
        JButton button = new JButton("file select");
        button.addActionListener(this);
        JPanel buttonPanel = new JPanel();
        buttonPanel.add(button);

        label = new JLabel();

        JPanel labelPanel = new JPanel();
        labelPanel.add(label);

        getContentPane().add(labelPanel, BorderLayout.CENTER);
        getContentPane().add(buttonPanel, BorderLayout.PAGE_END);
    }

    public void actionPerformed(ActionEvent e){
        JFileChooser filechooser = new JFileChooser();

        int selected = filechooser.showOpenDialog(this);
        if (selected == JFileChooser.APPROVE_OPTION){
            File file = filechooser.getSelectedFile();
            label.setText(file.getName());
        }else if (selected == JFileChooser.CANCEL_OPTION){
            label.setText("キャンセルされました");
        }else if (selected == JFileChooser.ERROR_OPTION){
            label.setText("エラー又は取消しがありました");
        }
    }
}
