package org.deeplearning4j.sake2vec;

import com.atilika.kuromoji.ipadic.Token;
import com.atilika.kuromoji.ipadic.Tokenizer;
import com.cignoir.cabocha.Cabocha;
import com.cignoir.node.Chunk;
import com.cignoir.node.Sentence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by b1012059 on 2015/11/14.
 */

public class Converter {

    public Converter(){

    }

    private static Logger log = LoggerFactory.getLogger(Converter.class);

    private Collection<String> wakatiSent(String sentence){
        Tokenizer tokenizer = new Tokenizer() ;
        List<Token> tokens = tokenizer.tokenize(sentence);
        ArrayList<String> ret = new ArrayList<String>();

        for (Token token : tokens) {
            String[] features = token.getAllFeaturesArray();
            System.out.print(token.getSurface()+" ");
            //ret.add(token.getSurface());

            if(features[0].equals("名詞")) {
                ret.add(token.getSurface());
            } else if(features[0].equals("形容詞")){
                ret.add(token.getSurface());
            }
        }

        System.out.println();
        //log.info("分かち書き:" + String.valueOf(ret));
        return ret;
    }

    private void cabochaSent(String sentence) {
        Cabocha cabocha = new Cabocha("C:\\Program Files\\CaboCha\\bin\\cabocha.exe");

        try {
            Sentence sent = cabocha.execute(sentence);
            List<Chunk> chunkList = sent.getChunks();

            for (Chunk chunk : chunkList) {
                List<com.cignoir.node.Token> tokens = chunk.getTokens();
                for (com.cignoir.node.Token token : tokens) {
                    System.out.println(token.getBase() + ": " + token.getPos());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public Collection<String> convWakati(String sentence){
        return this.wakatiSent(sentence);
    }

    public void convCaboCha(String sentence){
        this.cabochaSent(sentence);
    }

    private static void testConverter(){
        Converter conv = new Converter();
        //conv.convWakati("フルーティさが強いとそれが口に残ることも多いですが、獺祭は後味すっきり");
        //conv.convWakati("人間の双曲線関数的な報酬系の振る舞いが人間の意思を決定する。");
        //System.out.println(conv.Wakati("『報酬』の在り方を巡って、脳内にある報酬系のエージェント群が相互に意志によって選択されようとする過程が、葛藤や選択と呼ばれる。"));
        //System.out.println("名詞と形容詞:" + conv.convWakati("獺祭より甘くなくて辛い日本酒は？"));
        conv.convCaboCha("獺祭より甘くなくて辛い日本酒は？");
    }

    public static void main(String args[]){
        testConverter();
    }
}
