package org.deeplearning4j.sake2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.Collection;

/**
 * GUIに入力されたqueryをword2vecが
 * 理解できる形式に変換する
 * Created by b1012059 on 2015/08/11.
 * @author Wataru Matsudate
 */
public class SakeChanger extends SakeUtils {
    private static Logger log = LoggerFactory.getLogger(SakeChanger.class);
    private String fileName;//modelName;
    private boolean flag;
    private Sake2Vec2 vec;


    /**
     *
     * @param fileName
     * @param flag
     */
    public SakeChanger(String fileName, boolean flag){

        this.flag = flag;
        this.fileName = fileName;
        /*if(flag) this.modelName = fileName;
        else this.fileName = fileName;*/
    }

    /**
     *
     */
    public SakeChanger(){
    }

    /**
     *
     * @param sentence
     * @throws Exception
     */
    public void runSake2Vec(Collection<String> sentence) throws Exception{
        vec.runSake2vec2();
        sakeSimilarity(vec, sentence);
    }

    /**
     *
     * @param sentence
     * @return
     */
    public String transRun(String sentence){
        Converter conv = new Converter();
        conv.convWakati(sentence);
        /*if(flag) vec = new Sake2Vec2(modelName, flag);
        else vec = new Sake2Vec2(fileName, flag);*/

        return input(sentence);
    }

    /**
     *
     * @return
     * @throws Exception
     */
    public String output() throws Exception {
        vec = new Sake2Vec2(fileName, flag);
        //Converter conv = new Converter();
        //conv.convSentence(sentence);
        //runSake2Vec(conv.convSentence(sentence));
        //runSake2Vec(conv.convSentence(sentence));
        return calculate();
    }

    /**
     *
     * @param sentence
     * @return
     */
    private String input(String sentence){
        String result;
        result = ("あなた: " + sentence + "\n");
        return result;
    }

    /**
     *
     * @return
     */
    private String calculate(){
        String result;
        BigDecimal bi = new BigDecimal(String.valueOf(simResult));
        double su = bi.setScale(2,BigDecimal.ROUND_HALF_UP).doubleValue();
        result = ("sake2vec: 類似度は" + su + "です。" + judgment(su) + "。\n");
        return result;
    }

    /**
     * Test Method
     */
    private static void testSakeChanger(){
        System.out.println("Hello World!!");
    }

    public static void main(String args[]){
        testSakeChanger();
    }

}
