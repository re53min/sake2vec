package org.deeplearning4j.sake2vec;

import com.atilika.kuromoji.ipadic.Token;
import com.atilika.kuromoji.ipadic.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Sentenceに対する処理
 * Created by b1012059 on 2015/11/14.
 */

public class Converter {

    public Converter(){

    }

    private static Logger log = LoggerFactory.getLogger(Converter.class);

    /**
     * Sentenceを形態素解析して分かち書き
     * Kuromojiにて実装
     * @param sentence
     * @return
     */
    private ArrayList<String> wakatiSent(String sentence){
        Tokenizer tokenizer = new Tokenizer() ;
        List<Token> tokens = tokenizer.tokenize(sentence);
        ArrayList<String> ret = new ArrayList<String>();

        for (Token token : tokens) {
            String[] features = token.getAllFeaturesArray();
            System.out.print(token.getSurface()+" ");
            //ret.add(token.getSurface());

            if(features[0].equals("名詞")) {
                ret.add(token.getSurface());
            } else if(features[0].equals("形容詞")){
                ret.add(token.getSurface());
            }
        }

        System.out.println();
        //log.info("分かち書き:" + String.valueOf(ret));
        return ret;
    }


    /**
     * Sentenceを係り受け解析
     * CaboChaにて実装
     * @param sentence
     */
    private void cabochaSent(String sentence) {
        String cabocha = "C:\\Program Files\\CaboCha\\bin\\cabocha.exe";

        try{
            byte[] bytes = {-17, -69 , -65};

            String btmp = new String(bytes, "UTF-8");

            sentence = sentence.replace(btmp, "");

            ProcessBuilder pb = new ProcessBuilder(cabocha, "-f1");
            Process process = pb.start();

            OutputStreamWriter osw = new OutputStreamWriter(process.getOutputStream(),"UTF-8");
            osw.write(sentence);
            osw.close();

            InputStream is = process.getInputStream();
            BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));

            String Line;

            ArrayList<String> out = new ArrayList<>();

            while((Line = br.readLine())!= null){
                out.add(Line);

                //System.out.println(Line);
            }
            process.destroy();
            process.waitFor();

            hogehoge(out);

        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * hogehoge
     * @param out
     */
    private void hogehoge(ArrayList out){
        ArrayList<String> positive = new ArrayList<>();
        ArrayList<String> negative = new ArrayList<>();

        out.forEach(tmp -> System.out.println(tmp));
    }

    /**
     * 梯形しか受け付けないやつ
     */
    private void hogeSent(){

    }

    public ArrayList<String> convWakati(String sentence){
        return this.wakatiSent(sentence);
    }

    public void convCaboCha(String sentence){
        this.cabochaSent(sentence);
    }

    /**
     * Tester
     */
    private static void testConverter(){
        Converter conv = new Converter();
        System.out.println(conv.convWakati("『報酬』の在り方を巡って、脳内にある報酬系のエージェント群が相互に意志によって選択されようとする過程が、葛藤や選択と呼ばれる。"));
        //System.out.println("名詞と形容詞:" + conv.convWakati("人間の双曲線関数的な報酬系の振る舞いが人間の意思を決定する。"));
        //conv.convCaboCha("獺祭は甘くないことはない");
    }

    public static void main(String args[]){
        testConverter();
    }
}
