package org.deeplearning4j.sake2vec;

/**
 * sake2vec2本体
 * Created by b1012059 on 2015/04/25.
 * @auther Wataru Matsudate
 */

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

class Sake2Vec2 {
    private String fileName, modelName;
    private String word1;
    private String word2;
    private Word2Vec vec;
    private boolean flag;
    private static Logger log = LoggerFactory.getLogger(Sake2Vec2.class);


    /**
     *
     * @param fileName
     * @param word1
     * @param word2
     */
    public Sake2Vec2(String fileName, String word1, String word2) {

        this.fileName = fileName;
        this.word1 = word1;
        this.word2 = word2;

    }

    /**
     *
     * @param word1
     * @param word2
     */
    public Sake2Vec2(String word1, String word2){

        this.word1 = word1;
        this.word2 = word2;

    }

    /**
     *
     * @param fileName
     */
    public Sake2Vec2(String fileName, boolean flag){

        this.flag = flag;
        if(flag) this.modelName = fileName;
        else this.fileName = fileName;

    }

    /**
     *
     */
    public Sake2Vec2(){

    }

    /**
     *
     * @throws Exception
     */
    public void runSake2vec2() throws Exception {

        //googleNV = new File("GoogleNews-vectors-negative300.bin.gz");

        if(flag){
            //sakeCorpus = new File(modelName);
            log.info("Exist word vectors model. Reload model...");

            vec = WordVectorSerializer.loadFullModel(modelName);

            log.info("****************Reload model finished********************");

        } else {
            String filePath = new ClassPathResource(fileName).getFile().getAbsolutePath();

            log.info("Load & Vectorize Sentences....");
            // Strip white space before and after for each line
            SentenceIterator iter = UimaSentenceIterator.createWithPath(filePath);
            // Split on white spaces in the line to get words
            TokenizerFactory t = new DefaultTokenizerFactory();
            t.setTokenPreProcessor(new CommonPreprocessor());

            InMemoryLookupCache cache = new InMemoryLookupCache();
            WeightLookupTable table = new InMemoryLookupTable.Builder()
                    .vectorLength(100)
                    .useAdaGrad(false)
                    .cache(cache)
                    .lr(0.025f).build();

            log.info("Building model....");
            Word2Vec vec = new Word2Vec.Builder()
                    .minWordFrequency(5).iterations(1)
                    .layerSize(100).lookupTable(table)
                    .stopWords(new ArrayList<String>())
                    .vocabCache(cache).seed(42)
                    .windowSize(5).iterate(iter).tokenizerFactory(t).build();

            log.info("Fitting Word2Vec model....");
            vec.fit();

            //InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
            //table.getSyn0().diviRowVector(table.getSyn0().norm2(0));

            log.info("save model");
            WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");
            WordVectorSerializer.writeFullModel(vec, fileName);
            this.modelName = fileName;

            //ここにバグがあるよ
            this.flag = true;

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
    public double sakeSimilar(String word1,  String word2) throws Exception {
        double result = 0.0;
        double sim;
        if(vec != null){
            try {
                log.info("***********************vec exists***********************");
                //similarity(string　A, string　B):AとBの近似値
                sim = vec.similarity(word1, word2);
                result = sim;
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            try {
                log.info("***********************vec null***********************");
                runSake2vec2();
                sim = vec.similarity(word1, word2);
                result = sim;
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
        return this.sakeWordsNearest(Arrays.asList(new String[]{word}), new ArrayList(), number);
    }


    /**
     *
     * @param posi
     * @param nega
     * @param number
     * @return
     */
    public Collection<String> sakeWordsNearest(List<String> posi, List<String> nega, int number) throws Exception {
        Collection<String> result = null;
        Collection<String> similar;

        if(vec != null){
            try {
                //wordsNearest(string　A, int　N):Aに近い単語をN個抽出
                similar = vec.wordsNearest(posi, nega, number);
                //System.out.println(similar);
                result = similar;
            } catch (Exception e){
                e.printStackTrace();
            }
        } else {
            try {
                runSake2vec2();
                similar = vec.wordsNearest(posi, nega, number);
                result = similar;
            } catch (Exception e){
                e.printStackTrace();
            }
        }
        return result;
    }

}
