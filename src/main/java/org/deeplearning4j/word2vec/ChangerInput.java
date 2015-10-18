package org.deeplearning4j.word2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * GUIに入力されたqueryをword2vecが
 * 理解できる形式に変換する
 * Created by b1012059 on 2015/08/11.
 * @author Wataru Matsudate
 */
public class ChangerInput {
    private static Logger log = LoggerFactory.getLogger(Sake2Vec.class);
    private String fileName, modelName;
    private double simResult;
    private boolean flag;

    public ChangerInput(String fileName, boolean flag){

        this.flag = flag;
        if(flag) this.modelName = fileName;
        else this.fileName = fileName;
    }

    public ChangerInput(){
    }

    public String transRun(String sentence) throws Exception {
        Sake2Vec2 vec;
        if(flag) vec = new Sake2Vec2(modelName, flag);
        else vec = new Sake2Vec2(fileName, flag);

        vec.runSake2vec();

        return changeInput(vec, sentence);
    }

    public String output(){
        String result;
        result = changeOutput();
        return result;
    }

    private String changeInput(Sake2Vec2 vec, String sentence){
        String result;
        sake2vec2Run(vec, sentence);
        result = ("あなた: " + sentence + "\n");
        return result;
    }


    private String changeOutput(){
        String result;
        BigDecimal bi = new BigDecimal(String.valueOf(simResult));
        double su = bi.setScale(2,BigDecimal.ROUND_HALF_UP).doubleValue();
        result = ("sake2vec: 類似度は" + su + "です。" + judgment(su) + "。\n");
        return result;
    }

    private String judgment(double sim){
        String text;
        if(sim > 0.8){
            text = "かなり似ています";
        } else if(sim < 0.3){
            text = "それほど似ていません";
        } else {
            text = "まあまあ似ている…？";
        }
        return text;
    }

    private void sake2vec2Run(Sake2Vec2 vec, String sentence){
        List<String> posi = new ArrayList();
        List<String> nega = new ArrayList();

        String[] posiTmp = sentence.split(",", -1);
        for(int i = 0; i < posiTmp.length; i++){
            posi.add(posiTmp[i]);
            log.info("Sentence temp:" + posi.get(i));
        }

        //similarity of word1 and word2
        try {
            simResult = vec.sakeSimilar(posi.get(0), posi.get(1));
        } catch (Exception e) {
            e.printStackTrace();
        }
        log.info("Similarity between " + posi.get(0) + " and " + posi.get(1) + ": " + simResult);
        log.info("*********************************************************");
    }

}
