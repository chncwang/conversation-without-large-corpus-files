#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

#include <cmath>
#include <unordered_map>
#include <vector>
#include <array>
#include <set>
#include <string>
#include <memory>
#include <tuple>
#include <queue>
#include <algorithm>
#include <boost/format.hpp>
#include "insnet/insnet.h"
#include "fmt/include/fmt/core.h"
#include "tinyutf8.h"
#include "model_params.h"
#include "hyper_params.h"
#include "single_turn_conversation/print.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/default_config.h"

using namespace std;
using namespace insnet;

set<std::array<int, 3>> triSet(const vector<int> &sentence) {
    if (sentence.size() < 3) {
        cerr << "triSet" << endl;
        abort();
    }
    using std::array;
    set<array<int, 3>> results;
    for (int i = 0; i < sentence.size() - 2; ++i) {
        array<int, 3> tri = {sentence.at(i + 0), sentence.at(i + 1), sentence.at(i + 2)};
        results.insert(tri);
    }

    if (results.size() != sentence.size() - 2) {
        cerr << boost::format("triSet - result size is %1%, but sentence len is %2%") %
            results.size() % sentence.size() << endl;
        abort();
    }

    return results;
}

set<int> repeatedIds(const vector<int> &sentence) {
    auto tri_set = triSet(sentence);
    set<int> results;
    for (const auto &tri : tri_set) {
        int sentence_len = sentence.size();
        if (tri.at(0) == sentence.at(sentence_len - 2) &&
                tri.at(1) == sentence.at(sentence_len - 1)) {
            results.insert(tri.at(2));
        }
    }
    return results;
}

struct WordIdAndProbability {
    int dim;
    int word_id;
    dtype probability;

    WordIdAndProbability() = default;
    WordIdAndProbability(const WordIdAndProbability &word_id_and_probability) = default;
    WordIdAndProbability(int dimm, int wordid, dtype prob) : dim(dimm), word_id(wordid),
    probability(prob) {}
};

string getSentence(const vector<int> &word_ids_vector, const ModelParams &model_params) {
    string words;
    for (const int &w : word_ids_vector) {
        string str = model_params.lookup_table.vocab.from_id(w);
        words += str;
    }
    return words;
}

#define BEAM_SEARCH_KEY "beam_search"

string toSentenceStringWithSpace(const vector<string> &sentence) {
    string result;
    for (const string &w : sentence) {
        result += w + " ";
    }
    return result;
}

void printWordIds(const vector<WordIdAndProbability> &word_ids_with_probability_vector,
        const Embedding<Param> &lookup_table) {
    for (const WordIdAndProbability &ids : word_ids_with_probability_vector) {
        cout << lookup_table.vocab.from_id(ids.word_id);
    }
    cout << endl;
}

void printWordIdsWithKeywords(const vector<WordIdAndProbability> &word_ids_with_probability_vector,
        const Embedding<Param> &lookup_table,
        const unordered_map<string, float> &idf_table,
        bool print_space = false) {
    cout << "keywords:" << endl;
    for (int i = 0; i < word_ids_with_probability_vector.size(); i += 2) {
        cout << lookup_table.vocab.from_id(word_ids_with_probability_vector.at(i).word_id);
        cout << " " << word_ids_with_probability_vector.at(i).word_id << "  ";
    }
    cout << endl;
    cout << "words:" << endl;
    for (int i = 1; i < word_ids_with_probability_vector.size(); i += 2) {
        int word_id = word_ids_with_probability_vector.at(i).word_id;
        cout << lookup_table.vocab.from_id(word_id);
        if (print_space && i + 2 < word_ids_with_probability_vector.size()) {
            cout << " ";
        }
    }
    cout << endl;
}

int countNgramDuplicate(const vector<int> &ids, int n) {
    if (n >= ids.size()) {
        return 0;
    }
    vector<int> target;
    for (int i = 0; i < n; ++i) {
        target.push_back(ids.at(ids.size() - n + i));
    }

    int duplicate_count = 0;

    for (int i = 0; i < ids.size() - n; ++i) {
        bool same = true;
        for (int j = 0; j < n; ++j) {
            if (target.at(j) != ids.at(i + j)) {
                same = false;
                break;
            }
        }
        if (same) {
            ++duplicate_count;
        }
    }

    return duplicate_count;
}

//void updateBeamSearchResultScore(BeamSearchResult &beam_search_result,
//        const NgramPenalty& penalty) {
//    vector<WordIdAndProbability> word_id_and_probability = beam_search_result.getPath();
//    vector<int> ids = transferVector<int, WordIdAndProbability>(word_id_and_probability, [](
//                const WordIdAndProbability &a) {return a.word_id;});
//    dtype extra_score = 0.0f;
//    vector<dtype> penalties = {penalty.one, penalty.two, penalty.three};
//    std::array<int, 3> counts;
//    for (int i = 3; i > 0; --i) {
//        int duplicate_count = countNgramDuplicate(ids, i);
//        counts.at(i - 1) = duplicate_count;
//        extra_score -= penalties.at(i - 1) * duplicate_count;
//    }
//    beam_search_result.setExtraScore(beam_search_result.getExtraScore() + extra_score);
//    std::array<int, 3> original_counts = beam_search_result.ngramCounts();
//    std::array<int, 3> new_counts = {original_counts.at(0) + counts.at(0),
//        original_counts.at(1) = counts.at(1), original_counts.at(2) + counts.at(2)};
//    beam_search_result.setNgramCounts(new_counts);
//}

//bool beamSearchResultCmp(const BeamSearchResult &a, const BeamSearchResult &b) {
//    return a.finalScore() != a.finalScore() ?  a.finalScore() > b.finalScore();
//}

//vector<BeamSearchResult> mostProbableResults(
////        const vector<DecoderComponents> &beam,
//        const vector<BeamSearchResult> &last_results,
//        int current_word,
//        int k,
//        const ModelParams &model_params,
//        const DefaultConfig &default_config,
//        bool is_first,
//        bool check_tri_gram,
//        const vector<string> &black_list,
//        const unordered_map<string, float> &idf_table,
//        Graph &graph) {
//    vector<Node *> nodes;
//    int beam_i = 0;
//    for (const DecoderComponents &decoder_components : beam) {
//        auto path = last_results.at(beam_i).getPath();
//        if (path.size() % 2 == 0) {
//            cerr << "path is even" << endl;
//            abort();
//        }
//        Node *node = decoder_components.wordvector_to_onehots.at(current_word);
//        nodes.push_back(node);
//        ++beam_i;
//    }
//    if (nodes.size() != last_results.size() && !last_results.empty()) {
//        cerr << boost::format(
//                "nodes size is not equal to last_results size, nodes size is %1% but last_results size is %2%")
//            % nodes.size() % last_results.size() << endl;
//        abort();
//    }
//
//    auto cmp = [&](const BeamSearchResult &a, const BeamSearchResult &b) {
//        graph.addFLOPs(1, BEAM_SEARCH_KEY);
//        return a.finalScore() > b.finalScore();
//    };
//    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
//    vector<BeamSearchResult> results;
////    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
//    for (int i = 0; i < nodes.size(); ++i) {
//        const Node &node = *nodes.at(i);
//
//        BeamSearchResult beam_search_result;
//        set<int> repeated_ids;
//        if (check_tri_gram) {
//            vector<int> word_ids;
//            int m = 0;
//            for (const auto &e : last_results.at(i).getPath()) {
//                if (m++ % 2 == 0) {
//                    continue;
//                }
//                word_ids.push_back(e.word_id);
//            }
//            repeated_ids = repeatedIds(word_ids);
//        }
//        for (int j = 0; j < nodes.at(i)->size(); ++j) {
//            if (repeated_ids.find(j) != repeated_ids.end()) {
//                continue;
//            }
//            if (j == model_params.lookup_table.getElemId(insnet::UNKNOWN_WORD)) {
//                continue;
//            }
//            int stop_id = model_params.lookup_table.vocab.from_string(STOP_SYMBOL);
//            if (j == stop_id && !last_results.empty() &&
//                    last_results.at(i).getPath().back().word_id != stop_id) {
//                continue;
//            }
//            dtype value = node.getVal().v[j];
//            dtype log_probability = log(value);
//            graph.addFLOPs(1, BEAM_SEARCH_KEY);
//            dtype word_probability = value;
//            vector<WordIdAndProbability> word_ids;
//            if (!last_results.empty()) {
//                log_probability += last_results.at(i).finalLogProbability();
//                graph.addFLOPs(1, BEAM_SEARCH_KEY);
//                word_ids = last_results.at(i).getPath();
//            }
//            word_ids.push_back(WordIdAndProbability(node.size(), j, word_probability));
//            beam_search_result =  BeamSearchResult(beam.at(i), word_ids, log_probability);
//            graph.addFLOPs(1, BEAM_SEARCH_KEY);
//            int local_size = k;
//            if (queue.size() < local_size) {
//                queue.push(beam_search_result);
//            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
//                graph.addFLOPs(1, BEAM_SEARCH_KEY);
//                queue.pop();
//                queue.push(beam_search_result);
//            } else {
//                graph.addFLOPs(1, BEAM_SEARCH_KEY);
//            }
//        }
//    }
//
//    while (!queue.empty()) {
//        auto &e = queue.top();
//        results.push_back(e);
//        queue.pop();
//    }
//
//    vector<BeamSearchResult> final_results;
//    int i = 0;
//    for (const BeamSearchResult &result : results) {
//        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
//                [](const WordIdAndProbability &in) ->int {return in.word_id;});
//        string sentence = ::getSentence(ids, model_params);
//        final_results.push_back(result);
//        cout << boost::format("mostProbableResults - i:%1% prob:%2% score:%3%") % i %
//            result.finalLogProbability() % result.finalScore() << endl;
//        printWordIdsWithKeywords(result.getPath(), model_params.lookup_table, idf_table);
//        ++i;
//    }
//
//    return final_results;
//}

//vector<BeamSearchResult> mostProbableKeywords(
//        vector<DecoderComponents> &beam,
//        const vector<BeamSearchResult> &last_results,
//        const unordered_map<string ,float> word_idf_table,
//        int word_pos,
//        int k,
//        Graph &graph,
//        ModelParams &model_params,
//        const HyperParams &hyper_params,
//        const DefaultConfig &default_config,
//        bool is_first,
//        set<int> &searched_ids,
//        const vector<string> &black_list) {
//    cout << "black size:" << black_list.size() << endl;
//    vector<Node *> keyword_nodes, hiddens, nodes;
//    for (int ii = 0; ii < beam.size(); ++ii) {
//        bool should_predict_keyword;
//        if (last_results.empty()) {
//            should_predict_keyword = true;
//        } else {
//            vector<WordIdAndProbability> path = last_results.at(ii).getPath();
//            int size = path.size();
//            should_predict_keyword = path.at(size - 2).word_id == path.at(size - 1).word_id;
//        }
//        Node *node, *keyword_node, *hidden;
//        hidden = beam.at(ii).decoder._hiddens.at(word_pos);
//        if (should_predict_keyword) {
//            DecoderComponents &components = beam.at(ii);

//            if (components.decoder_lookups.size() != word_pos) {
//                cerr << boost::format("size:%1% word_pos:%2%") % components.decoder_lookups.size()
//                    % word_pos << endl;
//                abort();
//            }

//            Node *context_concated = insnet::cat({components.decoder._hiddens.at(word_pos),
//                    components.contexts.at(word_pos)});

//            Node *keyword = insnet::linear(graph, model_params.hidden_to_keyword_params,
//                    *context_concated);
//            keyword_node = keyword;

//            int last_keyword_id;
//            if (last_results.empty()) {
//                last_keyword_id = model_params.lookup_table.nVSize - 1;
//            } else {
//                vector<WordIdAndProbability> path = last_results.at(ii).getPath();
//                last_keyword_id = path.at(path.size() - 2).word_id;
//            }

//            Node *keyword_vector_to_onehot = linearWordVector(graph,
//                    last_keyword_id + 1, model_params.lookup_table.E, *keyword);
//            Node *softmax = softmax(graph, *keyword_vector_to_onehot);

//            components.keyword_vector_to_onehots.push_back(softmax);
//            node = softmax;
//        } else {
//            node = nullptr;
//            keyword_node = nullptr;
//        }
//        nodes.push_back(node);
//        keyword_nodes.push_back(keyword_node);
//        hiddens.push_back(hidden);
//    }
//    graph.compute();

//    auto cmp = [&](const BeamSearchResult &a, const BeamSearchResult &b) {
//        graph.addFLOPs(1, BEAM_SEARCH_KEY);
//        return a.finalScore() > b.finalScore();
//    };
//    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
//    vector<BeamSearchResult> results;
//    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
//        const Node *node_ptr = nodes.at(i);
//        if (node_ptr == nullptr) {
//            vector<WordIdAndProbability> new_id_and_probs = last_results.at(i).getPath();
//            WordIdAndProbability &last_keyword = new_id_and_probs.at(new_id_and_probs.size() - 2);
//            WordIdAndProbability &last_norm = new_id_and_probs.at(new_id_and_probs.size() - 1);
//            WordIdAndProbability w = {last_keyword.dim, last_keyword.word_id,
//                last_norm.probability};
//            new_id_and_probs.push_back(w);
//            BeamSearchResult beam_search_result(beam.at(i), new_id_and_probs,
//                    last_results.at(i).finalLogProbability());
//            graph.addFLOPs(1, BEAM_SEARCH_KEY);
//            if (queue.size() < k) {
//                queue.push(beam_search_result);
//            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
//                graph.addFLOPs(1, BEAM_SEARCH_KEY);
//                queue.pop();
//                queue.push(beam_search_result);
//            } else {
//                graph.addFLOPs(1, BEAM_SEARCH_KEY);
//            }
//        } else {
//            const Node &node = *nodes.at(i);

//            BeamSearchResult beam_search_result;
//            for (int j = 0; j < nodes.at(i)->getDim(); ++j) {
//                bool should_continue = false;
//                if (is_first) {
//                    if (searched_ids.find(j) != searched_ids.end()) {
//                        continue;
//                    }
//                }
//                if (j == model_params.lookup_table.getElemId(::unknownkey)) {
//                    continue;
//                }

//                if (word_idf_table.at(model_params.lookup_table.elems.from_id(j)) >= 9.0f) {
//                    break;
//                }

//                for (const string &black : black_list) {
//                    if (black == model_params.lookup_table.elems.from_id(j)) {
//                        should_continue = true;
//                    }
//                }
//                if (should_continue) {
//                    continue;
//                }
//                const string &word = model_params.lookup_table.elems.from_id(j);
//                if (word_pos == 0 && word_idf_table.at(word) <= default_config.keyword_bound) {
//                    continue;
//                }
//                dtype value = node.getVal().v[j];
//                dtype log_probability = log(value);
//                dtype word_probability = value;
//                vector<WordIdAndProbability> word_ids;
//                if (!last_results.empty()) {
//                    log_probability += last_results.at(i).finalLogProbability();
//                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
//                    word_ids = last_results.at(i).getPath();
//                }
//                if (log_probability != log_probability) {
//                    cerr << node.getVal().vec() << endl;
//                    cerr << "keyword node:" << endl << keyword_nodes.at(i)->getVal().vec() << endl;
//                    cerr << "hidden node:" << endl << hiddens.at(i)->getVal().vec() << endl;
//                    Json::StreamWriterBuilder builder;
//                    builder["commentStyle"] = "None";
//                    builder["indentation"] = "";
//                    string json_str = Json::writeString(builder, model_params.hidden_to_keyword_params.W.toJson());
//                    cerr << "param W:" << endl << json_str << endl;
//                    abort();
//                }
//                word_ids.push_back(WordIdAndProbability(node.getDim(), j, word_probability));

//                BeamSearchResult local = BeamSearchResult(beam.at(i), word_ids, log_probability);
//                graph.addFLOPs(1, BEAM_SEARCH_KEY);
//                if (queue.size() < k) {
//                    queue.push(local);
//                } else if (queue.top().finalScore() < local.finalScore()) {
//                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
//                    queue.pop();
//                    queue.push(local);
//                } else {
//                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
//                }
//            }
//        }
//    }

//    while (!queue.empty()) {
//        auto &e = queue.top();
//        if (e.finalScore() != e.finalScore()) {
//            printWordIdsWithKeywords(e.getPath(), model_params.lookup_table, word_idf_table);
//            cerr << "final score nan" << endl;
//            abort();
//        }
//        if (is_first) {
//            int size = e.getPath().size();
//            if (size != 1) {
//                cerr << boost::format("size is not 1:%1%\n") % size;
//                abort();
//            }
//            searched_ids.insert(e.getPath().at(0).word_id);
//        }
//        results.push_back(e);
//        queue.pop();
//    }

//    vector<BeamSearchResult> final_results;
//    int i = 0;
//    for (const BeamSearchResult &result : results) {
//        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
//                [](const WordIdAndProbability &in) ->int {return in.word_id;});
//        string sentence = ::getSentence(ids, model_params);
//        final_results.push_back(result);
//        cout << boost::format("mostProbableKeywords - i:%1% prob:%2% score:%3%") % i %
//            result.finalLogProbability() % result.finalScore() << endl;
//        printWordIdsWithKeywords(result.getPath(), model_params.lookup_table, word_idf_table);
//        ++i;
//    }

//    return final_results;
//}

struct GraphBuilder {
    Node *encoder_hiddens;
    int enc_len;

    void forward(Graph &graph, const vector<string> &sentence, const HyperParams &hyper_params,
            ModelParams &model_params) {
        vector<Node *> embs;
        for (const string &w : sentence) {
            Node *emb = embedding(graph, w, model_params.lookup_table);
            emb = dropout(*emb, hyper_params.dropout);
            embs.push_back(emb);
        }
        Node *h0 = tensor(graph, hyper_params.hidden_dim, 0);
        LSTMState initial_state = {h0, h0};
        std::vector<Node *> l2r = lstm(initial_state, embs, model_params.l2r_encoder_params,
                hyper_params.dropout);
        std::reverse(embs.begin(), embs.end());
        std::vector<Node *> r2l = lstm(initial_state, embs, model_params.r2l_encoder_params,
                hyper_params.dropout);
        std::reverse(r2l.begin(), r2l.end());

        Node *l2r_matrix = cat(l2r);
        Node *r2l_matrix = cat(r2l);
        encoder_hiddens = cat({l2r_matrix, r2l_matrix}, l2r.size());
        enc_len = l2r.size();
    }

    pair<vector<Node *>, vector<Node *>> forwardDecoder(Graph &graph, Node &enc,
            const std::vector<std::string> &answers,
            const std::vector<std::string> &keywords,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        int keyword_bound = model_params.lookup_table.nVSize;
        vector<Node *> keyword_probs, normal_probs;

        Node *h0 = insnet::tensor(graph, hyper_params.hidden_dim,0);
        LSTMState last_state = {h0, h0};
        for (int i = 0; i < answers.size(); ++i) {
            if (i > 0) {
                keyword_bound = model_params.lookup_table.vocab.from_string(keywords.at(i - 1)) +
                    1;
            }
            int normal_bound = model_params.lookup_table.vocab.from_string(keywords.at(i)) + 1;
            if (normal_bound > keyword_bound) {
                print(answers);
                print(keywords);
                abort();
            }
            Node *dec_normal_emb, *dec_keyword_emb, *guide;
            bool predict_keyword = i == 0 ||  answers.at(i - 1) == keywords.at(i - 1);
            if (i > 0) {
                const string &dec_input = answers.at(i - 1);
                dec_normal_emb = embedding(graph, dec_input, model_params.lookup_table);
                dec_normal_emb = dropout(*dec_normal_emb, hyper_params.dropout);

                if (predict_keyword) {
                    dec_keyword_emb = tensor(graph, hyper_params.word_dim, 0);
                } else {
                    const string &keyword = keywords.at(i - 1);
                    dec_keyword_emb = embedding(graph, keyword, model_params.lookup_table);
                    dec_keyword_emb = dropout(*dec_keyword_emb, hyper_params.dropout);
                }

                guide = last_state.hidden;
            } else {
                Node *bucket = tensor(graph, hyper_params.word_dim, 0);
                dec_normal_emb = bucket;
                dec_keyword_emb = bucket;
                guide = tensor(graph, hyper_params.hidden_dim, 0);
            }

            int col = enc.size() / 2 / hyper_params.hidden_dim;
            if (col * 2 * hyper_params.hidden_dim != enc.size()) {
                cerr << fmt::format("col:{} hidden_dim:{} enc size:{}", col,
                        hyper_params.hidden_dim, enc.size()) << endl;
                abort();
            }

            Node *context = additiveAttention(*guide, enc, col,
                    model_params.attention_params).first;
            Node *in = insnet::cat({dec_normal_emb, dec_keyword_emb, context});
            last_state = lstm(last_state, *in, model_params.decoder_params, hyper_params.dropout);
            Node *keyword_prob = nullptr;
            if (predict_keyword) {
                keyword_prob = insnet::cat({last_state.hidden, context});
                keyword_prob = insnet::linear(*keyword_prob,
                        model_params.hidden_to_keyword_params);
                keyword_prob = insnet::linear(*keyword_prob, model_params.lookup_table.E);
                keyword_prob = insnet::split(*keyword_prob, keyword_bound, 0);
                keyword_prob = insnet::softmax(*keyword_prob);
            }
            keyword_probs.push_back(keyword_prob);

            Node *output_keyword_emb = embedding(graph, keywords.at(i), model_params.lookup_table);
            Node *normal_word_prob = insnet::cat({last_state.hidden, context, dec_normal_emb,
                    output_keyword_emb});
            normal_word_prob = insnet::linear(*normal_word_prob,
                    model_params.hidden_to_wordvector_params_a);
            normal_word_prob = insnet::relu(*normal_word_prob);
            normal_word_prob = insnet::dropout(*normal_word_prob, hyper_params.dropout);
            normal_word_prob = insnet::linear(*normal_word_prob,
                    model_params.hidden_to_wordvector_params_b);
            normal_word_prob = insnet::linear(*normal_word_prob, model_params.lookup_table.E);
            if (i == answers.size() - 1) {
                normal_word_prob = insnet::split(*normal_word_prob, normal_bound, 0);
            } else if (i < answers.size() - 1) {
                normal_word_prob = insnet::split(*normal_word_prob, normal_bound - 1, 1);
            } else {
                cerr << "error state" << endl;
                abort();
            }
            normal_word_prob = insnet::softmax(*normal_word_prob);
            normal_probs.push_back(normal_word_prob);
        }
        return make_pair(move(keyword_probs), move(normal_probs));
    }
};

#endif
