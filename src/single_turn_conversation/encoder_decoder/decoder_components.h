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
    vector<LookupNode<Param> *> decoder_lookups_before_dropout;
    vector<DropoutNode *> decoder_lookups;
    vector<Node *> decoder_keyword_lookups;
    vector<Node *> decoder_to_wordvectors;
    vector<Node *> decoder_to_keyword_vectors;
    vector<Node *> wordvector_to_onehots;
    vector<Node *> keyword_vector_to_onehots;
    vector<DynamicLSTMBuilder> decoders;
    vector<Node*> contexts;

    DecoderComponents(int layer) {
        decoders.resize(layer);
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            Node &keyword_input,
            vector<Node *> &encoder_hiddens,
            bool is_training) {
        shared_ptr<AdditiveAttentionBuilder> attention_builder(new AdditiveAttentionBuilder);
        Node *guide;
        if (decoders.at(0).size() == 0) {
            guide = n3ldg_plus::bucket(graph, decoders.size() * hyper_params.hidden_dim, 0);
        } else {
            vector<Node*> ins;
            for (int i = 0; i < decoders.size(); ++i) {
                ins.push_back(decoders.at(i)._hiddens.back());
            }
            guide = n3ldg_plus::concat(graph, ins);
        }
        attention_builder->forward(graph, model_params.attention_params, encoder_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);

        vector<Node *> ins = {&input, &keyword_input, attention_builder->_hidden};
        Node *concat = n3ldg_plus::concat(graph, ins);

        decoders.at(0).forward(graph, *model_params.left_to_right_decoder_params.ptrs().at(0),
                *concat, *n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0),
                *n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0),
                hyper_params.dropout, is_training);
        Node *last_hidden = attention_builder->_hidden;

        for (int i = 1; i < decoders.size(); ++i) {
            last_hidden = n3ldg_plus::add(graph,
                    {last_hidden, decoders.at(i - 1)._hiddens.back()});
            ins = {&input, &keyword_input, last_hidden};
            concat = n3ldg_plus::concat(graph, ins);
            decoders.at(i).forward(graph, *model_params.left_to_right_decoder_params.ptrs().at(i),
                    *concat, *n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0),
                    *n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0),
                    hyper_params.dropout, is_training);
        }
    }

    ResultAndKeywordVectors decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            vector<Node *> &encoder_hiddens,
            int i,
            bool return_keyword) {
        vector<Node *> concat_inputs = {
            contexts.at(i), decoders.back()._hiddens.at(i),
            i == 0 ? n3ldg_plus::bucket(graph, hyper_params.word_dim, 0) :
                static_cast<Node*>(decoder_lookups.at(i - 1))
        };
        if (decoder_lookups.size() != i) {
            cerr << boost::format("decoder_lookups size:%1% i:%2%") % decoder_lookups.size() %
                i << endl;
            abort();
        }
        Node *concat_node = n3ldg_plus::concat(graph, concat_inputs);

        Node *keyword;
        if (return_keyword) {
            ConcatNode *context_concated = new ConcatNode;
            context_concated->init(2 * hyper_params.hidden_dim);
            context_concated->forward(graph, {decoders.back()._hiddens.at(i), contexts.at(i)});

            keyword = n3ldg_plus::linear(graph, model_params.hidden_to_keyword_params,
                    *context_concated);
        } else {
            keyword = nullptr;
        }

        ConcatNode *keyword_concated = new ConcatNode();
        keyword_concated->init(concat_node->getDim() + hyper_params.word_dim);
        keyword_concated->forward(graph, {concat_node, decoder_keyword_lookups.at(i)});

        Node *decoder_to_wordvector = n3ldg_plus::linear(graph,
                model_params.hidden_to_wordvector_params, *keyword_concated);

        return {decoder_to_wordvector, keyword};
    }
};

#endif
