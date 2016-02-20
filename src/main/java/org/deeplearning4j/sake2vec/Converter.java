package org.deeplearning4j.sake2vec;

import com.atilika.kuromoji.ipadic.Token;
import com.atilika.kuromoji.ipadic.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

/**
 * Sentenceに対する処理
 * Created by b1012059 on 2015/11/14.
 */

public class Converter {

    private static Logger log = LoggerFactory.getLogger(Converter.class);

    private Collection<String> positive;
    private Collection<String> negative;
    private ArrayList<String> ret;
    private List<Token> tokens;
    private Stack stack;
    private String query;


    public Converter(String query){
        this.query = query;
        this.positive = new ArrayList<>();
        this.negative = new ArrayList<>();
        this.ret = new ArrayList<>();
        this.stack = new Stack();

        if(query != null){
            Tokenizer tokenizer = new Tokenizer();
            this.tokens = tokenizer.tokenize(this.query);
        } else {
            log.info("Query is empty");
        }
    }

    /**
     * Sentenceを形態素解析して分かち書き
     * Kuromojiを使用
     * @return
     */
    private void wakatiSentence(){

        for (Token token : tokens) {
            String[] features = token.getAllFeaturesArray();

            //System.out.print(token.getSurface()+" ");
            //ret.add(token.getSurface());

            if(features[0].equals("名詞")) {
                this.ret.add(token.getSurface());
            } else if(features[0].equals("形容詞")){
                this.ret.add(token.getSurface());
            //} else if(features[0].equals("ない")){
            //    ret.add(token.getSurface());
            } else if(features[6].equals("、")) {
                this.ret.add(token.getSurface());
            }
        }

        //System.out.println();
        wordFormula(this.ret);
    }


    private void wordFormula(ArrayList<String> ret){

        ret.forEach(s -> {
           if (s.equals("、")) negative.add((String) stack.pop());
           else if(s.equals("ない")) negative.add((String) stack.pop());
           else stack.push(s);
        });

        positive = stack;
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

            wordFormula(out);

        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public Collection<String> getPositiveList(){
        return this.positive;
    }

    public Collection<String> getNegativeList(){
        return this.negative;
    }

    public void convSentence(){
        this.wakatiSentence();
    }

    public void convCaboCha(String sentence){
        this.cabochaSent(sentence);
    }

    /**
     * Tester
     */
    private static void testConverter(){
        //String sentence = "『報酬』の在り方を巡って、脳内にある報酬系のエージェント群が相互に意志によって選択されようとする過程が、葛藤や選択と呼ばれる。";
        String sentence = "フルーティ";
        Converter conv = new Converter(sentence);
        conv.convSentence();

        System.out.println("positive words: " + conv.positive);
        System.out.println("---------------------------------");
        System.out.println("negative words: " + conv.negative);
    }

    public static void main(String args[]){
        testConverter();
    }
}
