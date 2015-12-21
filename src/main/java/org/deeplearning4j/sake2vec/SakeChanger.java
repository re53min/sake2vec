package org.deeplearning4j.sake2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;

import static org.deeplearning4j.sake2vec.SakeUtils.judgment;
import static org.deeplearning4j.sake2vec.SakeUtils.sakeSimilarity;

/**
 * GUIに入力されたqueryをword2vecが
 * 理解できる形式に変換する
 * Created by b1012059 on 2015/08/11.
 * @author Wataru Matsudate
 */
public class SakeChanger {
    private static Logger log = LoggerFactory.getLogger(SakeChanger.class);
    private String fileName;
    private boolean flag;
    private Sake2Vec2 vec;
    private Converter conv;
    private ArrayList<String> ret;


    /**
     *
     * @param fileName
     * @param flag
     */
    public SakeChanger(String fileName, boolean flag){

        this.flag = flag;
        this.fileName = fileName;
        this.vec = new Sake2Vec2(this.fileName, this.flag);
        this.conv = new Converter();
        this.ret = new ArrayList<>();
    }

    /**
     *
     */
    public SakeChanger(){
    }

    /**
     *
     * @throws Exception
     */
    public void runSake2Vec()throws Exception{
        log.info("Starting Run Sake2Vec");
        vec.runSake2vec2();
    }

    /**
     *
     * @param sentence
     * @return
     */
    public String input(String sentence){
        String str = "あなた:" + sentence;
        log.info("Starting Convert Sentences");
        ret = conv.convWakati(sentence);

        return str;
    }

    /**
     *
     * @return
     * @throws Exception
     */
    public String output() throws Exception {
        runSake2Vec();
        log.info("Starting Calculate Words");
        return calculate(sakeSimilarity(vec, ret));
    }

    /**
     *
     * @return
     */
    private String calculate(double similarity){
        BigDecimal bi = new BigDecimal(String.valueOf(similarity));
        double su = bi.setScale(2,BigDecimal.ROUND_HALF_UP).doubleValue();
        String result = ("sake2vec: 類似度は" + su + "です。" + judgment(su) + "。\n");

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