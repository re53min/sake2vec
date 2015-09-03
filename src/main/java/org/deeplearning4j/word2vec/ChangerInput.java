package org.deeplearning4j.word2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * GUIに入力されたsentenceをword2vecが
 * 理解できる形式に変換する
 * Created by b1012059 on 2015/08/11.
 * @author Wataru Matsudate
 */
public class ChangerInput {
    private static Logger log = LoggerFactory.getLogger(Sake2Vec.class);
    private Sake2Vec2 vec;
    private String fileName;
    private double simResult;

    public ChangerInput(String fileName){
        this.fileName = fileName;
    }

    public ChangerInput(){
    }

    public String transRun(String sentence) throws Exception {
        if(fileName != null) {
            vec = new Sake2Vec2(fileName);
        } else {
            vec = new Sake2Vec2();
        }
        vec.Sake2vecExample();
        return changeInput(sentence);
    }

    public String output(){
        String result;
        result = changeOutput();
        return result;
    }

    private String changeInput(String sentence){
        String result;
        sake2vec2Run(sentence);
        result = ("私:" + sentence + "\n");
        return result;
    }


    private String changeOutput(){
        String result;
        BigDecimal bi = new BigDecimal(String.valueOf(simResult));
        double su = bi.setScale(2,BigDecimal.ROUND_HALF_UP).doubleValue();
        result = ("sake2vec:" + su + "です。" + judgment(su) + "。\n");
        return result;
    }

    private String judgment(double sim){
        String text;
        if(sim > 8){
            text = "かなり似ています";
        } else if(sim <3){
            text = "それほど似ていません";
        } else {
            text = "似てはいます";
        }
        return text;
    }

    private void sake2vec2Run(String sentence){
        List<String> posi = new ArrayList();
        List<String> nega = new ArrayList();

        String[] posiTmp = sentence.split(",", -1);
        for(int i = 0; i < posiTmp.length; i++){
            posi.add(posiTmp[i]);
            log.info("Sentence temp:" + posi.get(i));
        }

        //similarity of word1 and word2
        try {
            simResult = vec.sake2vecSimilarity(posi.get(0), posi.get(1));
        } catch (Exception e) {
            e.printStackTrace();
        }
        log.info("Similarity between " + posi.get(0) + " and " + posi.get(1) + ": " + simResult);
    }

}
