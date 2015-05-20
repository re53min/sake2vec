package org.deeplearning4j.word2vec;

/**
 * Created by b1012059 on 2015/04/25.
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

public class Sake2vec {
    String fileName;
    String word1;
    String word2;


    public Sake2vec(String fileName, String word1, String word2) {

        this.fileName = fileName;
        this.word1 = word1;
        this.word2 = word2;

    }

    public double Sake2vecExample() throws Exception {

        double sim;


        ClassPathResource resource = new ClassPathResource(this.fileName);
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

        Word2Vec vec = new Word2Vec.Builder().sampling(1e-5)
                .minWordFrequency(5).batchSize(1000).useAdaGrad(false).layerSize(layerSize)
                .iterations(3).learningRate(0.025).minLearningRate(1e-2).negativeSample(10)
                .iterate(iter).tokenizerFactory(t).build();
        vec.fit();

        //similarity(string　A, string　B):AとBの近似値
        sim = vec.similarity(word1, word2);
        System.out.println("Similarity between people and money " + sim);

        //wordsNearest(string　A, int　N):Aに近い単語をN個抽出
        Collection<String> similar = vec.wordsNearest(word1, 20);
        System.out.println(similar);

        /*Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(200).useAdaGrad(false)
                .normalize(false).usePca(false).build();


        vec.lookupTable().plotVocab(tsne);*/


    return sim;
    }

    /*

     */
    public String sake2vecResult() {
        String result = "デバッグ用";
        try {
            double temp = Sake2vecExample();
            result = word1 + "と" + word2 + "の類似値は" + String.valueOf(temp);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }
}