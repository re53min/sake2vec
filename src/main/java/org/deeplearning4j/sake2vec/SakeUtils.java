package org.deeplearning4j.sake2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Created by b1012059 on 2015/10/21.
 */
public class SakeUtils extends Sake2Vec2{
    private static Logger log = LoggerFactory.getLogger(SakeUtils.class);

    public static String judgment(double sim){
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

    /**
     *
     * @param vec
     * @param sentence
     */
    public static double sakeSimilarity(Sake2Vec2 vec, Collection<String> sentence){
        double simResult = 0.0;
        List<String> posi = new ArrayList();
        List<String> nega = new ArrayList();

        /*String[] posiTmp = sentence.split(",", -1);
        for(int i = 0; i < posiTmp.length; i++){
            posi.add(posiTmp[i]);
            log.info("Sentence temp:" + posi.get(i));
        }*/

        for(String tmp : sentence){
            posi.add(tmp);
            log.info("Sentence add:" + tmp);
        }

        //similarity of word1 and word2
        try {
            simResult = vec.sakeSimilar(posi.get(0), posi.get(1));
        } catch (Exception e) {
            e.printStackTrace();
        }
        log.info("Similarity between " + posi.get(0) + " and " + posi.get(1) + ": " + simResult);
        log.info("*********************************************************");

        return simResult;
    }

    public static HashMap<String, Double> sakeNearest(Sake2Vec2 vec, String word, int number) throws Exception {
        String[] sake = {"獺祭", "久保田", "八海山", "黒龍", "飛露喜", "田酒", "出羽桜", "〆張鶴", "蓬莱泉", "天狗舞"};
        String[] sakeCategory = {"薫酒", "爽酒", "醇酒", "熟酒"};
        String[] similar = vec.sakeWordsNearest(word, number).toArray(new String[0]);
        HashMap<String, Double> map = new HashMap<>();
        List<HashMap.Entry> entries = new ArrayList<HashMap.Entry>(map.entrySet());

        for(int i = 0; i < sake.length; i++){
            map.put(sake[i], vec.sakeSimilar(sake[i], similar[0]));
        }

        Collections.sort(entries, new Comparator(){
            public int compare(Object o1, Object o2){
                HashMap.Entry e1 =(HashMap.Entry)o1;
                HashMap.Entry e2 =(HashMap.Entry)o2;
                return ((Integer)e1.getValue()).compareTo((Integer)e2.getValue());
            }
        });



        return map;
    }

    /**
     * コサイン類似度
     * cosθ = (vectorA*vectorB) / (normA*normB)
     * 分子はベクトルの内積、分母はそれぞれのノルム
     * @param vectorA
     * @param vectorB
     * @return
     */
    public static double cosineSimilarity(double[] vectorA, double[] vectorB){
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for(int i = 0; i < vectorA.length; i++){
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }

        return dotProduct /(Math.sqrt(normA) * Math.sqrt(normB));
    }


    private static void testSakeUtils(){
        SakeUtils sake = new SakeUtils();
        Sake2Vec2 vec = new Sake2Vec2("test-words.txt", true);

        try {
            vec.runSake2vec2();
            System.out.println(sake.sakeNearest(vec, "甘い", 3));
        } catch (Exception e) {
            e.printStackTrace();
        }

        //System.out.println("cosine similarity is " + sake.cosineSimilarity(new double[]{1.0, 0, 0, 0}, new double[]{1.0, 0, 0, 0}));
    }

    public static void main(String args[]){
        testSakeUtils();
    }

}
