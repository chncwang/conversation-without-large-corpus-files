#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct DecoderComponents {
    std::vector<Node *> decoder_lookups;
    std::vector<Node *> decoder_to_wordvectors;
    std::vector<Node *> wordvector_to_onehots;
    DynamicLSTMBuilder decoder;
    vector<Node*> contexts;

    BucketNode *bucket(int dim, Graph &graph) {
        BucketNode *node(new BucketNode);
        node->init(dim);
        node->forward(graph, 0);
        return node;
    }

    void forward(Graph &graph, const HyperParams &hyper_params, LSTM1Params &decoder_params,
            AdditiveAttentionParams &attention_params,
            Node &input,
            vector<Node *> &encoder_hiddens,
            bool is_training) {
        shared_ptr<AdditiveAttentionBuilder> attention_builder(new AdditiveAttentionBuilder);
        Node *guide = decoder.size() == 0 ?
            static_cast<Node*>(bucket(hyper_params.hidden_dim,
                        graph)) : static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));
        attention_builder->forward(graph, attention_params, encoder_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);

        vector<Node *> ins = {&input, attention_builder->_hidden};
        Node *concat = n3ldg_plus::concat(graph, ins);

        decoder.forward(graph, decoder_params, *concat, *bucket(hyper_params.hidden_dim, graph),
                *bucket(hyper_params.hidden_dim, graph), hyper_params.dropout, is_training);
    }

    Node *decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            UniParams &to_word_params,
            int i) {
        vector<Node *> concat_inputs = {
            contexts.at(i), decoder._hiddens.at(i),
            i == 0 ? bucket(to_word_params.W.inDim() - 2 * hyper_params.hidden_dim, graph) :
                static_cast<Node*>(decoder_lookups.at(i - 1))
        };
        if (decoder_lookups.size() != i + 1) {
            cerr << boost::format("decoder_lookups size:%1% i:%2%") % decoder_lookups.size() %
                i << endl;
            abort();
        }
        Node *concat_node = n3ldg_plus::concat(graph, concat_inputs);
        Node *decoder_to_wordvector = n3ldg_plus::linear(graph, to_word_params, *concat_node);

        return decoder_to_wordvector;
    }
};

#endif
