#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct ResultAndKeywordVectors {
    Node *result;
    Node *keyword;
};

struct DecoderComponents {
    std::vector<Node *> decoder_to_wordvectors;
    std::vector<Node *> decoder_to_keyword_vectors;
    std::vector<Node *> wordvector_to_onehots;
    std::vector<Node *> keyword_vector_to_onehots;
    std::vector<Node *> decoder_lookups;
    DynamicLSTMBuilder decoder;
    vector<Node*> contexts;

    BucketNode *bucket(int dim, Graph &graph) {
        BucketNode *node(new BucketNode);
        node->init(dim);
        node->forward(graph, 0);
        return node;
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            vector<Node *> &encoder_hiddens,
            bool is_training) {
        shared_ptr<AdditiveAttentionBuilder> attention_builder(new AdditiveAttentionBuilder);
        Node *guide = decoder.size() == 0 ?
            static_cast<Node*>(bucket(hyper_params.hidden_dim,
                        graph)) : static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));
        attention_builder->forward(graph, model_params.attention_params, encoder_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);

        vector<Node *> ins = {&input, attention_builder->_hidden};
        Node *concat = n3ldg_plus::concat(graph, ins);

        decoder.forward(graph, model_params.left_to_right_decoder_params, *concat,
                *bucket(hyper_params.hidden_dim, graph), *bucket(hyper_params.hidden_dim, graph),
                hyper_params.dropout, is_training);
    }

    Node *decoderToKeywordVector(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            vector<Node *> &encoder_hiddens,
            int i) {
        vector<Node *> concat_inputs = {
            contexts.at(i), decoder._hiddens.at(i)
        };
        Node *concat_node = n3ldg_plus::concat(graph, concat_inputs);
        return n3ldg_plus::linear(graph, model_params.hidden_to_keyword_params, *concat_node);
    }

    Node *decoderToNormalWordVector(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            vector<Node *> &encoder_hiddens,
            Node &keyword_embedding,
            int i) {
        Node *last_embedding = i == 0 ? n3ldg_plus::bucket(graph, hyper_params.word_dim, 0) :
            decoder_lookups.at(i - 1);
        vector<Node *> concat_inputs = {
            contexts.at(i), decoder._hiddens.at(i), &keyword_embedding, last_embedding
        };
        Node *concated = n3ldg_plus::concat(graph, concat_inputs);
        Node *normal_word_vector = n3ldg_plus::linear(graph,
                model_params.hidden_to_wordvector_params, *concated);
        normal_word_vector = n3ldg_plus::relu(graph, *normal_word_vector);
        return normal_word_vector;
    }
};

#endif
