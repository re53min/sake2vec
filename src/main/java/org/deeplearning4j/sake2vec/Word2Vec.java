package org.deeplearning4j.sake2vec;

/**
 * sake2vec2本体
 * Created by b1012059 on 2015/04/25.
 * @auther Wataru Matsudate
 */

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

public class Word2Vec {
    private String fileName, modelName;
    private org.deeplearning4j.models.word2vec.Word2Vec vec;
    private boolean flag;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);


    /**
     *
     * @param fileName
     */
    public Word2Vec(String fileName, boolean flag){

        this.flag = flag;
        if(flag) this.modelName = fileName;
        else this.fileName = fileName;

    }

    /**
     *
     */
    public Word2Vec(){

    }

    /**
     *
     * @throws Exception
     */
    public void runSake2vec2() throws Exception {

        if(flag){

            log.info("Exist word vectors model. Reload model...");

            vec = WordVectorSerializer.loadFullModel(modelName);

            log.info("****************Reload model finished********************");

        } else {
            String filePath = new org.canova.api.util.ClassPathResource(fileName).getFile().getAbsolutePath();

            log.info("Load & Vectorize Sentences....");
            // Strip white space before and after for each line
            SentenceIterator iter = new BasicLineIterator(filePath);
            // Split on white spaces in the line to get words
            TokenizerFactory t = new DefaultTokenizerFactory();
            t.setTokenPreProcessor(new CommonPreprocessor());

            log.info("Building model....");
            org.deeplearning4j.models.word2vec.Word2Vec vec = new org.deeplearning4j.models.word2vec.Word2Vec.Builder()
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

            log.info("save model");
            String str = fileName + "_model.txt";
            WordVectorSerializer.writeWordVectors(vec, str);
            //WordVectorSerializer.writeFullModel(vec, fileName);
            modelName = str;

            //ここにバグがあるよ
            flag = true;

            log.info("****************Build model finished********************");

        }
    }


    /**
     *
     * @param word1
     * @param word2
     * @return
     * @throws Exception
     */
    public double sakeSimilar(String word1,  String word2) {
        double result = 0.0;
        double sim;

        if(vec != null){
            try {
                log.info("***********************vec exists***********************");
                //similarity(string　A, string　B):AとBの近似値
                result = vec.similarity(word1, word2);
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            try {
                log.info("***********************vec null***********************");
                runSake2vec2();
                result = vec.similarity(word1, word2);
            } catch (Exception e){
                e.printStackTrace();
            }
        }
        return result;
    }

    /**
     *
     * @param word
     * @param number
     * @return
     * @throws Exception
     */
    public Collection<String> sakeWordsNearest(String word, int number) throws Exception{
        return this.sakeWordsNearest(Arrays.asList(word), new ArrayList(), number);
    }


    /**
     *
     * @param positive
     * @param negative
     * @param number
     * @return
     */
    public Collection<String> sakeWordsNearest(Collection<String> positive,
                                               Collection<String> negative, int number) {
        Collection<String> result = null;

        if(vec != null){
            try {
                log.info("***********************vec exists***********************");
                //wordsNearest(string　A, int　N):Aに近い単語をN個抽出
                result = vec.wordsNearest(positive, negative, number);
            } catch (Exception e){
                e.printStackTrace();
            }
        } else {
            try {
                log.info("***********************vec null***********************");
                runSake2vec2();
                result = vec.wordsNearest(positive, negative, number);
            } catch (Exception e){
                e.printStackTrace();
            }
        }
        return result;
    }

}
