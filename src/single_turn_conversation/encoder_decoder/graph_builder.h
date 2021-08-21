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
#include "single_turn_conversation/encoder_decoder/decoder_components.h"

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

class BeamSearchResult {
    public:
        BeamSearchResult() {}
        BeamSearchResult(const BeamSearchResult &beam_search_result) = default;
        BeamSearchResult(const DecoderCellComponents &decoder_components,
                const vector<WordIdAndProbability> &pathh,
                dtype log_probability) : decoder_components_(decoder_components), path_(pathh),
        final_log_probability(log_probability) {
        }

        dtype finalScore() const {
            set<int> unique_words;
            if (path_.size() % 2 == 0) {
                for (int i = 1; i < path_.size(); i += 2) {
                    const auto &e = path_.at(i);
                    unique_words.insert(e.word_id);
                }
            } else {
                for (const auto &e : path_) {
                    unique_words.insert(e.word_id);
                }
            }
            return final_log_probability/ unique_words.size();
        }

        dtype finalLogProbability() const {
            return final_log_probability;
        }

        vector<WordIdAndProbability> getPath() const {
            return path_;
        }

        DecoderCellComponents &decoderComponents() {
            return decoder_components_;
        }

    private:
        DecoderCellComponents decoder_components_;
        vector<WordIdAndProbability> path_;
        dtype final_log_probability;
};

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

vector<BeamSearchResult> mostProbableResults(
        const vector<DecoderCellComponents> &beam,
        const vector<BeamSearchResult> &last_results,
        int current_word,
        int k,
        const ModelParams &model_params,
        const DefaultConfig &default_config,
        bool is_first,
        bool check_tri_gram,
        const vector<string> &black_list,
        const unordered_map<string, float> &idf_table,
        Graph &graph) {
    vector<Node *> nodes;
    int beam_i = 0;
    for (const DecoderCellComponents &decoder_components : beam) {
        auto path = last_results.at(beam_i).getPath();
        if (path.size() % 2 == 0) {
            cerr << "path is even" << endl;
            abort();
        }
        Node *node = decoder_components.normal_prob;
        nodes.push_back(node);
        ++beam_i;
    }
    if (nodes.size() != last_results.size() && !last_results.empty()) {
        cerr << boost::format(
                "nodes size is not equal to last_results size, nodes size is %1% but last_results size is %2%")
            % nodes.size() % last_results.size() << endl;
        abort();
    }

    auto cmp = [&](const BeamSearchResult &a, const BeamSearchResult &b) {
        graph.addFLOPs(1, BEAM_SEARCH_KEY);
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;
    for (int i = 0; i < nodes.size(); ++i) {
        const Node &node = *nodes.at(i);

        BeamSearchResult beam_search_result;
        set<int> repeated_ids;
        if (check_tri_gram) {
            vector<int> word_ids;
            int m = 0;
            for (const auto &e : last_results.at(i).getPath()) {
                if (m++ % 2 == 0) {
                    continue;
                }
                word_ids.push_back(e.word_id);
            }
            repeated_ids = repeatedIds(word_ids);
        }
        for (int j = 0; j < nodes.at(i)->size(); ++j) {
            int shifted_id = beam.at(i).last_keyword_id == 0 ? j : j + 1;
            if (repeated_ids.find(shifted_id) != repeated_ids.end()) {
                continue;
            }
            if (shifted_id == model_params.lookup_table.getElemId(insnet::UNKNOWN_WORD)) {
                continue;
            }
            int stop_id = model_params.lookup_table.vocab.from_string(STOP_SYMBOL);
            if (shifted_id == stop_id && !last_results.empty() &&
                    last_results.at(i).getPath().back().word_id != stop_id) {
                continue;
            }
            dtype value = node.getVal().v[j];
            dtype log_probability = log(value);
            graph.addFLOPs(1, BEAM_SEARCH_KEY);
            dtype word_probability = value;
            vector<WordIdAndProbability> word_ids;
            if (!last_results.empty()) {
                log_probability += last_results.at(i).finalLogProbability();
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                word_ids = last_results.at(i).getPath();
            }
            word_ids.push_back(WordIdAndProbability(node.size(), shifted_id, word_probability));
            beam_search_result =  BeamSearchResult(beam.at(i), word_ids, log_probability);
            graph.addFLOPs(1, BEAM_SEARCH_KEY);
            int local_size = k;
            if (queue.size() < local_size) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                queue.pop();
                queue.push(beam_search_result);
            } else {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
            }
        }
    }

    while (!queue.empty()) {
        auto &e = queue.top();
        results.push_back(e);
        queue.pop();
    }

    vector<BeamSearchResult> final_results;
    int i = 0;
    for (const BeamSearchResult &result : results) {
        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
                [](const WordIdAndProbability &in) ->int {return in.word_id;});
        string sentence = ::getSentence(ids, model_params);
        final_results.push_back(result);
        cout << boost::format("mostProbableResults - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << endl;
        printWordIdsWithKeywords(result.getPath(), model_params.lookup_table, idf_table);
        ++i;
    }

    return final_results;
}

vector<BeamSearchResult> mostProbableKeywords(
        vector<DecoderCellComponents> &beam,
        const vector<BeamSearchResult> &last_results,
        const unordered_map<string ,float> word_idf_table,
        int word_pos,
        int k,
        Graph &graph,
        ModelParams &model_params,
        const HyperParams &hyper_params,
        const DefaultConfig &default_config,
        bool is_first,
        set<int> &searched_ids,
        const vector<string> &black_list) {
    cout << "black size:" << black_list.size() << endl;
    vector<Node *> nodes;
    for (int ii = 0; ii < beam.size(); ++ii) {
        bool should_predict_keyword;
        if (last_results.empty()) {
            should_predict_keyword = true;
        } else {
            vector<WordIdAndProbability> path = last_results.at(ii).getPath();
            int size = path.size();
            should_predict_keyword = path.at(size - 2).word_id == path.at(size - 1).word_id;
        }
        Node *node;
        if (should_predict_keyword) {
            DecoderCellComponents &components = beam.at(ii);

            int last_keyword_id;
            if (last_results.empty()) {
                last_keyword_id = model_params.lookup_table.nVSize - 1;
            } else {
                vector<WordIdAndProbability> path = last_results.at(ii).getPath();
                last_keyword_id = path.at(path.size() - 2).word_id;
            }

            Node *context_concated = insnet::cat({components.state.hidden, components.context});
            Node *keyword = insnet::linear(*context_concated,
                    model_params.hidden_to_keyword_params);
            keyword = insnet::linear(*keyword, model_params.lookup_table.E);
            keyword = insnet::split(*keyword, last_keyword_id + 1, 0);
            keyword = insnet::softmax(*keyword);

            node = keyword;
        } else {
            node = nullptr;
        }

        nodes.push_back(node);
    }
    graph.forward();

    auto cmp = [&](const BeamSearchResult &a, const BeamSearchResult &b) {
        graph.addFLOPs(1, BEAM_SEARCH_KEY);
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;
    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
        const Node *node_ptr = nodes.at(i);
        if (node_ptr == nullptr) {
            vector<WordIdAndProbability> new_id_and_probs = last_results.at(i).getPath();
            WordIdAndProbability &last_keyword = new_id_and_probs.at(new_id_and_probs.size() - 2);
            WordIdAndProbability &last_norm = new_id_and_probs.at(new_id_and_probs.size() - 1);
            WordIdAndProbability w = {last_keyword.dim, last_keyword.word_id,
                last_norm.probability};
            new_id_and_probs.push_back(w);
            BeamSearchResult beam_search_result(beam.at(i), new_id_and_probs,
                    last_results.at(i).finalLogProbability());
            graph.addFLOPs(1, BEAM_SEARCH_KEY);
            if (queue.size() < k) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                queue.pop();
                queue.push(beam_search_result);
            } else {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
            }
        } else {
            const Node &node = *nodes.at(i);

            BeamSearchResult beam_search_result;
            for (int j = 0; j < nodes.at(i)->size(); ++j) {
                bool should_continue = false;
                if (is_first) {
                    if (searched_ids.find(j) != searched_ids.end()) {
                        continue;
                    }
                }
                if (j == model_params.lookup_table.getElemId(UNKNOWN_WORD)) {
                    continue;
                }

//                if (word_idf_table.at(model_params.lookup_table.elems.from_id(j)) >= 9.0f) {
//                    break;
//                }

                if (should_continue) {
                    continue;
                }
//                const string &word = model_params.lookup_table.elems.from_id(j);
//                if (word_pos == 0 && word_idf_table.at(word) <= default_config.keyword_bound) {
//                    continue;
//                }
                dtype value = node.getVal().v[j];
                dtype log_probability = log(value);
                dtype word_probability = value;
                vector<WordIdAndProbability> word_ids;
                if (!last_results.empty()) {
                    log_probability += last_results.at(i).finalLogProbability();
                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
                    word_ids = last_results.at(i).getPath();
                }
                word_ids.push_back(WordIdAndProbability(node.size(), j, word_probability));

                BeamSearchResult local = BeamSearchResult(beam.at(i), word_ids, log_probability);
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                if (queue.size() < k) {
                    queue.push(local);
                } else if (queue.top().finalScore() < local.finalScore()) {
                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
                    queue.pop();
                    queue.push(local);
                } else {
                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
                }
            }
        }
    }

    while (!queue.empty()) {
        auto &e = queue.top();
        if (e.finalScore() != e.finalScore()) {
            printWordIdsWithKeywords(e.getPath(), model_params.lookup_table, word_idf_table);
            cerr << "final score nan" << endl;
            abort();
        }
        if (is_first) {
            int size = e.getPath().size();
            if (size != 1) {
                cerr << boost::format("size is not 1:%1%\n") % size;
                abort();
            }
            searched_ids.insert(e.getPath().at(0).word_id);
        }
        results.push_back(e);
        queue.pop();
    }

    vector<BeamSearchResult> final_results;
    int i = 0;
    for (const BeamSearchResult &result : results) {
        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
                [](const WordIdAndProbability &in) ->int {return in.word_id;});
        string sentence = ::getSentence(ids, model_params);
        final_results.push_back(result);
        cout << boost::format("mostProbableKeywords - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << endl;
        printWordIdsWithKeywords(result.getPath(), model_params.lookup_table, word_idf_table);
        ++i;
    }

    return final_results;
}

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
                keyword_prob = insnet::logSoftmax(*keyword_prob);
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
            normal_word_prob = insnet::logSoftmax(*normal_word_prob);
            normal_probs.push_back(normal_word_prob);
        }
        return make_pair(move(keyword_probs), move(normal_probs));
    }

    void stepDecoderHidden(Graph &graph, DecoderCellComponents &decoder_components,
            int i,
            const int *answer,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool predict_keyword) {
        Node *last_input, *last_keyword;
        if (i > 0) {
            Node *emb = insnet::embedding(graph, *answer, model_params.lookup_table.E);
            emb = insnet::dropout(*emb, hyper_params.dropout);
            last_input = emb;
            if (predict_keyword) {
                last_keyword = insnet::tensor(graph, hyper_params.word_dim, 0);
            } else {
                last_keyword = decoder_components.keyword_emb;
            }
        } else {
            Node *bucket = insnet::tensor(graph, hyper_params.word_dim, 0);
            last_input = bucket;
            last_keyword = bucket;
        }

        Node *guide = decoder_components.state.hidden;
        int col = encoder_hiddens->size() / 2 / hyper_params.hidden_dim;
        Node *context = additiveAttention(*guide, *encoder_hiddens, col,
                model_params.attention_params).first;
        decoder_components.context = context;
        Node *in = insnet::cat({last_input, last_keyword, context});
        decoder_components.state = lstm(decoder_components.state, *in, model_params.decoder_params,
                hyper_params.dropout);
    }

//    void forwardDecoderResultByOneStep(Graph &graph, DecoderCellComponents &decoder_components,
//            int i,
//            const string &keyword,
//            const HyperParams &hyper_params,
//            ModelParams &model_params) {
//        Node *keyword_lookup = embedding(graph, keyword, model_params.lookup_table);
//        keyword_lookup = dropout(*keyword_lookup, hyper_params.dropout);
//        decoder_components.keyword_emb = keyword_lookup;
//    }

    void stepNormalWordProbs(Graph &graph, DecoderCellComponents &decoder_components,
            int i,
            int keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            Node &encoder_hiddens) {
        Node *keyword_embedding = insnet::embedding(graph, keyword, model_params.lookup_table.E);
        decoder_components.keyword_emb = keyword_embedding;
        Node *last_normal_emb = i == 0 ? insnet::tensor(graph, hyper_params.word_dim, 0.0f) :
            decoder_components.normal_emb;
        Node *prob = insnet::cat({decoder_components.state.hidden, decoder_components.context,
                last_normal_emb, decoder_components.keyword_emb});
        prob = insnet::linear(*prob,
                model_params.hidden_to_wordvector_params_a);
        prob = insnet::relu(*prob);
        prob = insnet::dropout(*prob, hyper_params.dropout);
        prob = insnet::linear(*prob,
                model_params.hidden_to_wordvector_params_b);
        prob = insnet::linear(*prob, model_params.lookup_table.E);
        prob = keyword == 0 ? insnet::split(*prob, 1, 0) :
            insnet::split(*prob, keyword, 1);
        prob = insnet::softmax(*prob);
        decoder_components.normal_prob = prob;
    }

    pair<vector<WordIdAndProbability>, dtype> forwardDecoderUsingBeamSearch(Graph &graph,
            vector<DecoderCellComponents> &decoder_components_beam,
            const unordered_map<string, float> &word_idf_table,
            int k,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const DefaultConfig &default_config,
            const vector<string> &black_list) {
        Node *h0 = tensor(graph, hyper_params.hidden_dim, 0.0f);
        for (auto &c : decoder_components_beam) {
            c.state = {h0, h0};
        }

        vector<pair<vector<WordIdAndProbability>, dtype>> word_ids_result;
        vector<BeamSearchResult> most_probable_results;
        vector<int> last_answers, last_keywords;
        set<int> searched_ids;
        bool succeeded = false;

        auto beam = decoder_components_beam;
        cout << boost::format("beam size:%1%\n") % beam.size();

        for (int iter = 0; ; ++iter) {
            cout << "iter:" << iter << endl;
            most_probable_results.clear();
            auto beam = decoder_components_beam;
            int closured_count = word_ids_result.size();
            if (closured_count >= default_config.result_count_factor * k) {
                break;
            }

            for (int i = 0;; ++i) {
                cout << boost::format("forwardDecoderUsingBeamSearch i:%1%\n") % i;
                if (word_ids_result.size() >= default_config.result_count_factor * k ||
                        i > default_config.cut_length) {
                    break;
                }

                for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                    DecoderCellComponents &decoder_components = beam.at(beam_i);
                    bool predict_keyword = i == 0 ||
                        decoder_components.last_word_id == decoder_components.last_keyword_id;
                    stepDecoderHidden(graph, decoder_components, i,
                            i == 0 ? nullptr : &last_answers.at(beam_i), hyper_params,
                            model_params, predict_keyword);
                }
                cout << "forwardDecoderHiddenByOneStep:" << endl;
                graph.forward();

                most_probable_results = mostProbableKeywords(beam, most_probable_results,
                        word_idf_table, i, k, graph, model_params, hyper_params,
                        default_config, i == 0, searched_ids, black_list);
                for (auto &r : most_probable_results) {
                    int word_id = r.getPath().back().word_id;
                    r.decoderComponents().last_keyword_id = word_id;
                    if (word_id != 0) {
                        r.decoderComponents().normal_emb = insnet::embedding(graph, word_id,
                                model_params.lookup_table.E);
                    }
                }
                cout << "forwardDecoderResultByOneStep:" << endl;
                graph.forward();
                beam.clear();
                vector<BeamSearchResult> search_results;

                for (int j = 0; j < most_probable_results.size(); ++j) {
                    BeamSearchResult &beam_search_result = most_probable_results.at(j);
                    const vector<WordIdAndProbability> &word_ids = beam_search_result.getPath();
                    int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                    search_results.push_back(beam_search_result);
                    last_keywords.push_back(last_word_id);
                    beam.push_back(beam_search_result.decoderComponents());
                }
                most_probable_results = search_results;

                int beam_i = 0;
                for (auto &decoder_components : beam) {
                    stepNormalWordProbs(graph, decoder_components, i, last_keywords.at(beam_i),
                            hyper_params, model_params, *encoder_hiddens);
                    ++beam_i;
                }
                last_keywords.clear();
                cout << "forwardDecoderKeywordByOneStep:" << endl;
                graph.forward();
                cout << "forwardDecoderKeywordByOneStep: graph computed" << endl;

                last_answers.clear();
                most_probable_results = mostProbableResults(beam, most_probable_results, i,
                        k, model_params, default_config, i == 0, i>=3, black_list, word_idf_table,
                        graph);
                cout << boost::format("most_probable_results size:%1%") %
                    most_probable_results.size() << endl;
                beam.clear();
                vector<BeamSearchResult> stop_removed_results;
                int j = 0;
                for (BeamSearchResult &beam_search_result : most_probable_results) {
                    const vector<WordIdAndProbability> &word_ids = beam_search_result.getPath();

                    int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                    if (last_word_id == 0) {
                        word_ids_result.push_back(make_pair(word_ids,
                                    beam_search_result.finalScore()));
                        succeeded = true;
                    } else {
                        stop_removed_results.push_back(beam_search_result);
                        last_answers.push_back(last_word_id);
                        beam_search_result.decoderComponents().last_word_id = last_word_id;
                        beam.push_back(beam_search_result.decoderComponents());
                    }
                    ++j;
                }
                most_probable_results = stop_removed_results;

                if (beam.empty()) {
                    cout << boost::format("break for beam empty\n");
                    break;
                }
            }
        }

        if (word_ids_result.size() < default_config.result_count_factor * k) {
            cerr << boost::format("word_ids_result size is %1%, but beam_size is %2%") %
                word_ids_result.size() % k << endl;
            abort();
        }

        cout << endl<< "final search results:" << endl;
        for (const auto &pair : word_ids_result) {
            const vector<WordIdAndProbability> ids = pair.first;
            cout << boost::format("beam result:%1%") % exp(pair.second) << endl;
            printWordIdsWithKeywords(ids, model_params.lookup_table, word_idf_table);
        }

        auto compair = [](const pair<vector<WordIdAndProbability>, dtype> &a,
                const pair<vector<WordIdAndProbability>, dtype> &b) {
            return a.second < b.second;
        };
        auto max = max_element(word_ids_result.begin(), word_ids_result.end(), compair);

        return make_pair(max->first, exp(max->second));
    }
};

#endif
