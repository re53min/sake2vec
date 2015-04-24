package org.deeplearning4j.glove;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.glove.CoOccurrences;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class GloveExample {


    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });


        InMemoryLookupCache cache = new InMemoryLookupCache();
        TokenizerFactory t = new DefaultTokenizerFactory();
        TextVectorizer  tfidf = new TfidfVectorizer.Builder()
                .cache(cache).iterate(iter).minWords(1).tokenize(t).build();
        tfidf.fit();
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        t.setTokenPreProcessor(preProcessor);

        int layerSize = 300;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        CoOccurrences c = new CoOccurrences.Builder()
                .cache(cache).iterate(iter).tokenizer(t).build();

        c.fit();

        GloveWeightLookupTable table = new GloveWeightLookupTable.Builder()
                .cache(cache).lr(0.005).build();

        Glove vec = new Glove.Builder().learningRate(0.005).batchSize(1000)
                .cache(cache).coOccurrences(c).cache(cache)
                .iterations(30).vectorizer(tfidf).weights(table)
                .layerSize(layerSize).iterate(iter).tokenizer(t).minWordFrequency(1).symmetric(true)
                .windowSize(15).build();
        vec.fit();



        Collection<String> similar = vec.wordsNearest("day",20);
        System.out.println(similar);

        //vec.plotTsne();
        Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(500).useAdaGrad(false)
                .normalize(false).usePca(false).build();
        vec.lookupTable().plotVocab(tsne);
        //Word2VecLoader.writeTsneFormat(vec, tsne.getY(), new File("coords.csv"));



    }

}
