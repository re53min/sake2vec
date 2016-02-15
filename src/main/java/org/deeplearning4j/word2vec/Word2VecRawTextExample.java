package org.deeplearning4j.word2vec;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class Word2VecRawTextExample {

    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

    public static void main(String[] args) throws Exception {

        String filePath = new ClassPathResource("corpusB.txt").getFile().getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(filePath);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(2)
                .iterations(1)
                .layerSize(50)
                .seed(42)
                .windowSize(15)
                .learningRate(0.025)
                .minLearningRate(1e-2)
                .negativeSample(15)
                .sampling(1e-2)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        // Write word vectors
        WordVectorSerializer.writeWordVectors(vec, "corpus_B_Model.txt");

        //log.info("Closest Words:");
        //Collection<String> lst = vec.wordsNearest("day", 10);
        //System.out.println(lst);

        log.info("Word Formula");
        /*-----------------Q1-----------------
                日本酒Ａ-味わい表現+日本酒B = XXX

        log.info("---------------------Start Q1---------------------");
        Collection<String> Q1_1 = vec.wordsNearest(Arrays.asList("甘味", "久保田"), Arrays.asList("獺祭"), 5);
        System.out.println(Q1_1);
        Collection<String> Q1_2 = vec.wordsNearest(Arrays.asList("酸味", "八海山"), Arrays.asList("久保田"), 5);
        System.out.println(Q1_2);
        Collection<String> Q1_3 = vec.wordsNearest(Arrays.asList("酸味", "黒龍"), Arrays.asList("八海山"), 5);
        System.out.println(Q1_3);
        Collection<String> Q1_4 = vec.wordsNearest(Arrays.asList("甘味", "飛露喜"), Arrays.asList("黒龍"), 5);
        System.out.println(Q1_4);
        Collection<String> Q1_5 = vec.wordsNearest(Arrays.asList("旨味", "田酒"), Arrays.asList("飛露喜"), 5);
        System.out.println(Q1_5);
        Collection<String> Q1_6 = vec.wordsNearest(Arrays.asList("旨味", "天狗舞"), Arrays.asList("田酒"), 5);
        System.out.println(Q1_6);
        Collection<String> Q1_7 = vec.wordsNearest(Arrays.asList("酸味", "蓬莱泉"), Arrays.asList("天狗舞"), 5);
        System.out.println(Q1_7);
        Collection<String> Q1_8 = vec.wordsNearest(Arrays.asList("旨味", "出羽桜"), Arrays.asList("蓬莱泉"), 5);
        System.out.println(Q1_8);
        Collection<String> Q1_9 = vec.wordsNearest(Arrays.asList("甘味", "〆張鶴"), Arrays.asList("出羽桜"), 5);
        System.out.println(Q1_9);
        Collection<String> Q1_10 = vec.wordsNearest(Arrays.asList("酸味", "獺祭"), Arrays.asList("〆張鶴"), 5);
        System.out.println(Q1_10);
        log.info("---------------------End Q1---------------------");


        /*-----------------Q2-----------------
                日本酒Ａ-分類名+日本酒B = XXX
         *
        log.info("---------------------Start Q2---------------------");
        Collection<String> Q2_1 = vec.wordsNearest(Arrays.asList("薫酒", "久保田"), Arrays.asList("獺祭"), 5);
        System.out.println(Q2_1);
        Collection<String> Q2_2 = vec.wordsNearest(Arrays.asList("爽酒", "八海山"), Arrays.asList("久保田"), 5);
        System.out.println(Q2_2);
        Collection<String> Q2_3 = vec.wordsNearest(Arrays.asList("爽酒", "黒龍"), Arrays.asList("八海山"), 5);
        System.out.println(Q2_3);
        Collection<String> Q2_4 = vec.wordsNearest(Arrays.asList("爽酒", "飛露喜"), Arrays.asList("黒龍"), 5);
        System.out.println(Q2_4);
        Collection<String> Q2_5 = vec.wordsNearest(Arrays.asList("薫酒", "田酒"), Arrays.asList("飛露喜"), 5);
        System.out.println(Q2_5);
        Collection<String> Q2_6 = vec.wordsNearest(Arrays.asList("醇酒", "天狗舞"), Arrays.asList("田酒"), 5);
        System.out.println(Q2_6);
        Collection<String> Q2_7 = vec.wordsNearest(Arrays.asList("醇酒", "蓬莱泉"), Arrays.asList("天狗舞"), 5);
        System.out.println(Q2_7);
        Collection<String> Q2_8 = vec.wordsNearest(Arrays.asList("薫酒", "出羽桜"), Arrays.asList("蓬莱泉"), 5);
        System.out.println(Q2_8);
        Collection<String> Q2_9 = vec.wordsNearest(Arrays.asList("爽酒", "〆張鶴"), Arrays.asList("出羽桜"), 5);
        System.out.println(Q2_9);
        Collection<String> Q2_10 = vec.wordsNearest(Arrays.asList("爽酒", "獺祭"), Arrays.asList("〆張鶴"), 5);
        System.out.println(Q2_10);
        log.info("---------------------End Q2---------------------");

        /*-----------------Q3-----------------
                分類名A-味わい表現+分類名B = XXX
         *
        log.info("---------------------Start Q3---------------------");
        Collection<String> Q3_1 = vec.wordsNearest(Arrays.asList("甘味", "爽酒"), Arrays.asList("薫酒"), 5);
        System.out.println(Q3_1);
        Collection<String> Q3_2 = vec.wordsNearest(Arrays.asList("酸味", "醇酒"), Arrays.asList("爽酒"), 5);
        System.out.println(Q3_2);
        Collection<String> Q3_3 = vec.wordsNearest(Arrays.asList("雑味", "熟酒"), Arrays.asList("醇酒"), 5);
        System.out.println(Q3_3);
        Collection<String> Q3_4 = vec.wordsNearest(Arrays.asList("旨味", "薫酒"), Arrays.asList("熟酒"), 5);
        System.out.println(Q3_4);
        log.info("---------------------End Q3---------------------");


        /*-----------------Q4-----------------
                日本酒A-日本酒B+日本酒C = XXX
        */

        /*log.info("---------------------Start Q4---------------------");
        Collection<String> Q4_1 = vec.wordsNearest(Arrays.asList("出羽桜", "〆張鶴"), Arrays.asList("獺祭"), 5);
        System.out.println(Q4_1);
        Collection<String> Q4_2 = vec.wordsNearest(Arrays.asList("〆張鶴", "出羽桜"), Arrays.asList("八海山"), 5);
        System.out.println(Q4_2);
        //Collection<String> Q4_3 = vec.wordsNearest(Arrays.asList("天狗舞", "久保田"), Arrays.asList("飛露喜"), 5);
        //System.out.println(Q4_3);
        //Collection<String> Q4_4 = vec.wordsNearest(Arrays.asList("田酒", "飛露喜"), Arrays.asList("久保田"), 5);
        //System.out.println(Q4_4);
        log.info("---------------------End Q4---------------------");


        /*-----------------Q5-----------------
                日本酒A-日本酒B+分類名C = XXX
        *
        log.info("---------------------Start Q5---------------------");
        Collection<String> Q5_1 = vec.wordsNearest(Arrays.asList("八海山", "出羽桜"), Arrays.asList("獺祭"), 5);
        System.out.println(Q5_1);
        Collection<String> Q5_2 = vec.wordsNearest(Arrays.asList("獺祭", "〆張鶴"), Arrays.asList("八海山"), 5);
        System.out.println(Q5_2);
        //Collection<String> Q5_3 = vec.wordsNearest(Arrays.asList("久保田", "田酒"), Arrays.asList("飛露喜"), 5);
        //System.out.println(Q5_3);
        //Collection<String> Q5_4 = vec.wordsNearest(Arrays.asList("飛露喜", "天狗舞"), Arrays.asList("久保田"), 5);
        //System.out.println(Q5_4);
        log.info("---------------------End Q5---------------------");

        /*-----------------Q6-----------------
                味わい表現-味わい表現+味わい表現 = XXX
        *
        log.info("---------------------Start Q6---------------------");
        Collection<String> Q6_1 = vec.wordsNearest(Arrays.asList("酸味", "雑味"), Arrays.asList("甘味"), 5);
        System.out.println(Q6_1);
        Collection<String> Q6_2 = vec.wordsNearest(Arrays.asList("旨味", "甘味"), Arrays.asList("雑味"), 5);
        System.out.println(Q6_2);
        Collection<String> Q6_3 = vec.wordsNearest(Arrays.asList("甘味", "旨味"), Arrays.asList("酸味"), 5);
        System.out.println(Q6_3);
        Collection<String> Q6_4 = vec.wordsNearest(Arrays.asList("雑味", "酸味"), Arrays.asList("旨味"), 5);
        System.out.println(Q6_4);
        log.info("---------------------End Q6---------------------");

        */

        /*-----------------Q7-----------------
                味わい表現-味わい表現+味わい表現 = XXX
        *
        log.info("---------------------Start Q7---------------------");
        Collection<String> Q7_1 = vec.wordsNearest(Arrays.asList("爽酒", "熟酒"), Arrays.asList("薫酒"), 5);
        System.out.println(Q7_1);
        Collection<String> Q7_2 = vec.wordsNearest(Arrays.asList("醇酒", "薫酒"), Arrays.asList("熟酒"), 5);
        System.out.println(Q7_2);
        Collection<String> Q7_3 = vec.wordsNearest(Arrays.asList("薫酒", "醇酒"), Arrays.asList("爽酒"), 5);
        System.out.println(Q7_3);
        Collection<String> Q7_4 = vec.wordsNearest(Arrays.asList("熟酒", "爽酒"), Arrays.asList("醇酒"), 5);
        System.out.println(Q7_4);
        log.info("---------------------End Q7---------------------");
        */

        /*System.out.println(vec.similarity("酸味","爽酒"));
        System.out.println(vec.similarity("酸味","八海山"));
        System.out.println(vec.similarity("酸味","久保田"));
        System.out.println(vec.similarity("雑味","天狗舞"));
        System.out.println(vec.similarity("雑味","熟酒"));*/
        //System.out.println(vec.similarity("久保田","八海山"));
    }
}