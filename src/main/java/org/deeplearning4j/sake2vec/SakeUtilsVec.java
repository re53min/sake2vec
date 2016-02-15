package org.deeplearning4j.sake2vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by b1012059 on 2015/10/21.
 */
public class SakeUtilsVec extends Word2Vec {

    private static Logger log = LoggerFactory.getLogger(SakeUtilsVec.class);

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
     * @param word1
     * @param word2
     * @return
     */
    public static double sakeSimilarity(Word2Vec vec, String word1, String word2){
        double simResult = 0.0;

        //similarity of word1 and word2
        try {
            simResult = vec.sakeSimilar(word1, word2);
        } catch (Exception e) {
            e.printStackTrace();
        }
        log.info("Similarity between " + word1 + " and " + word2 + ": " + simResult);
        log.info("*********************************************************");

        return simResult;
    }

    public static Collection<String> sakeNearest(Word2Vec vec, Collection<String> positive,
                                                 Collection<String> negative, int number){
        Collection<String> result = null;

        //similarity of word1 and word2
        try {
            result = vec.sakeWordsNearest(positive, negative, number);
        } catch (Exception e) {
            e.printStackTrace();
        }
        log.info("Results: " + result);
        log.info("*********************************************************");

        return result;
    }


    public static ArrayList<String> sakeNearest(Word2Vec vec, String word) throws Exception {
        String[] sake = {"獺祭", "久保田", "八海山", "黒龍", "飛露喜", "田酒", "出羽桜", "〆張鶴", "蓬莱泉", "天狗舞"};
        String[] sakeCategory = {"薫酒", "爽酒", "醇酒", "熟酒"};
        ArrayList<String> result = new ArrayList<>();
        HashMap<String, Double> map = new HashMap<>();
        List<HashMap.Entry> entries = new ArrayList<>(map.entrySet());

        for(int i = 0; i < sake.length; i++){
            map.put(sake[i], sakeSimilarity(vec, sake[i], word));
        }

        Collections.sort(entries, new Comparator(){
            public int compare(Object o1, Object o2){
                HashMap.Entry e1 =(HashMap.Entry)o1;
                HashMap.Entry e2 =(HashMap.Entry)o2;
                return ((Integer)e1.getValue()).compareTo((Integer)e2.getValue());
            }
        });

        result.addAll(map.keySet().stream().collect(Collectors.toList()));

        return result;
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


    private static void testSakeUtils() {
        Word2Vec vec = new Word2Vec("words.txt", true);

        try {
            vec.runSake2vec2();
            System.out.println(sakeNearest(vec, Arrays.asList("加賀"), Arrays.asList("おっぱい"), 3));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]){
        testSakeUtils();
    }

}
