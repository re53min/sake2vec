package org.deeplearning4j.sake2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;

import static org.deeplearning4j.sake2vec.SakeUtilsVec.*;

/**
 * GUIに入力されたqueryをword2vecが
 * 理解できる形式に変換する
 * Created by b1012059 on 2015/08/11.
 * @author Wataru Matsudate
 */
public class Sake2Vec {
    private static Logger log = LoggerFactory.getLogger(Sake2Vec.class);
    private String fileName;
    private String query;
    private boolean flag;
    private Word2Vec vec;
    private Converter conv;

    /**
     *
     * @param query
     * @param fileName
     * @param flag
     */
    public Sake2Vec(String query, String fileName, boolean flag){
        log.info(fileName);

        this.query = query;
        this.fileName = fileName;
        this.flag = flag;
        this.vec = new Word2Vec(this.fileName, this.flag);
        this.conv = new Converter(query);

        this.conv.convSentence();
    }

    /**
     *
     * @return
     * @throws Exception
     */
    private ArrayList<String> output(int mode) throws Exception {
        ArrayList<String> result = new ArrayList<>();

        log.info("Starting Output Method!");
        if(vec != null) {
            vec.runWord2Vec();
        }

        switch (mode){
            case 0:
                // 演算結果のみ
                result = (ArrayList<String>) sakeNearest(vec, conv.getPositiveList(), conv.getNegativeList(), 5);
            case 1:
                // 演算結果から最も近い日本酒名
                ArrayList<String> formula = (ArrayList<String>) sakeNearest(vec,
                        conv.getPositiveList(), conv.getNegativeList(), 5);
                result = sakeNearest(vec, formula.get(0));
            case 2:
                // 類似度のみ
                result.add(calculate(sakeSimilarity(vec,
                        conv.getPositiveList().toString(), conv.getNegativeList().toString())));
        }

        return result;
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


    public String getQuery(){

        return this.query;
    }
    public ArrayList<String> getResult() throws Exception {
        return  this.output(1);
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