package org.deeplearning4j.word2vec;

/**
 * sake2vec本体
 * Created by b1012059 on 2015/04/25.
 * @auther b1012059 Wataru Matsudate
 */
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Collection;
import java.util.List;

class Sake2Vec {
    private String fileName;
    private String word1;
    private String word2;
    private Word2Vec vec;


    public Sake2Vec(String fileName, String word1, String word2) {

        this.fileName = fileName;
        this.word1 = word1;
        this.word2 = word2;

    }

    public Sake2Vec(String word1, String word2){

        this.word1 = word1;
        this.word2 = word2;

    }

    public Sake2Vec(String fileName){

        this.fileName = fileName;

    }

    public void Sake2vecExample() throws Exception {

        ClassPathResource resource = new ClassPathResource(fileName);
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        //DocumentIterator iter = new FileDocumentIterator(resource.getFile());
        TokenizerFactory t = new DefaultTokenizerFactory();
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        t.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d", "d");
                if (base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });

        int layerSize = 300;

        vec = new Word2Vec.Builder().sampling(1e-5)
                .minWordFrequency(5).batchSize(1000).useAdaGrad(false).layerSize(layerSize)
                .iterations(3).learningRate(0.025).minLearningRate(1e-2).negativeSample(10)
                .iterate(iter).tokenizerFactory(t).build();
        vec.fit();

        Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(200).useAdaGrad(false)
                .normalize(false).usePca(false).build();


        //vec.lookupTable();//.plotVocab(tsne);

    }


    public double sake2vecSimilarity() throws Exception {
        double result = 0.0;
        if(vec != null){
            try {
                //similarity(string　A, string　B):AとBの近似値
                double sim = vec.similarity(word1, word2);
                System.out.println("Similarity between people and money " + sim);
                result = sim;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    public Collection<String> sake2vecWordsNearest(int number){
        Collection<String> result = null;

        if(vec != null){
            //wordsNearest(string　A, int　N):Aに近い単語をN個抽出
            Collection<String> similar = vec.wordsNearest(word1, number);
            System.out.println(similar);
            result = similar;

            for(int i = 0; i < similar.size(); i++){
                List<String> tmpData = (List<String>) similar;
                double sim2 = vec.similarity(word1, tmpData.get(i));
                System.out.println(word1 + " and " + tmpData.get(i) + " is " + sim2);
            }
        }
        return result;
    }
}