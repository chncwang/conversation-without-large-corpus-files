#include "cxxopts.hpp"
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <random>
#include "INIReader.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include <boost/format.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include "N3LDG.h"
#include "single_turn_conversation/data_manager.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/bleu.h"
#include "single_turn_conversation/perplex.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/encoder_decoder/graph_builder.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"

using namespace std;
using namespace cxxopts;
using namespace boost::asio;
using boost::is_any_of;
using boost::format;
using boost::filesystem::path;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;

unordered_map<string, float> calculateIdf(const vector<vector<string>> sentences) {
    cout << "sentences size:" << sentences.size() << endl;
    unordered_map<string, int> doc_counts;
    int i = 0;
    for (const vector<string> &sentence : sentences) {
        if (i++ % 10000 == 0) {
            cout << i << " ";
        }
        set<string> words;
        for (const string &word : sentence) {
            words.insert(word);
        }

        for (const string &word : words) {
            auto it = doc_counts.find(word);
            if (it == doc_counts.end()) {
                doc_counts.insert(make_pair(word, 1));
            } else {
                ++doc_counts.at(word);
            }
        }
    }
    cout << endl;

    unordered_map<string, float> result;
    for (const auto &it : doc_counts) {
        float idf = log(sentences.size() / static_cast<float>(it.second));
        if (idf < 0.0) {
            cerr << "idf:" << idf << endl;
            abort();
        }
        result.insert(make_pair(it.first, idf));
    }

    return result;
}

void addWord(unordered_map<string, int> &word_counts, const string &word) {
    auto it = word_counts.find(word);
    if (it == word_counts.end()) {
        word_counts.insert(make_pair(word, 1));
    } else {
        it->second++;
    }
}

void addWord(unordered_map<string, int> &word_counts, const vector<string> &sentence) {
    for (const string &word : sentence) {
        addWord(word_counts, word);
    }
}

DefaultConfig parseDefaultConfig(INIReader &ini_reader) {
    DefaultConfig default_config;
    static const string SECTION = "default";
    default_config.train_pair_file = ini_reader.Get(SECTION, "train_pair_file", "");
    if (default_config.train_pair_file.empty()) {
        cerr << "train pair file empty" << endl;
        abort();
    }
    default_config.dev_pair_file = ini_reader.Get(SECTION, "dev_pair_file", "");
    if (default_config.dev_pair_file.empty()) {
        cerr << "dev pair file empty" << endl;
        abort();
    }
    default_config.test_pair_file = ini_reader.Get(SECTION, "test_pair_file", "");
    if (default_config.test_pair_file.empty()) {
        cerr << "test pair file empty" << endl;
        abort();
    }

    default_config.post_file = ini_reader.Get(SECTION, "post_file", "");
    if (default_config.post_file.empty()) {
        cerr << "post file empty" << endl;
        abort();
    }

    default_config.response_file = ini_reader.Get(SECTION, "response_file", "");
    if (default_config.post_file.empty()) {
        cerr << "post file empty" << endl;
        abort();
    }

    string program_mode_str = ini_reader.Get(SECTION, "program_mode", "");
    ProgramMode program_mode;
    if (program_mode_str == "interacting") {
        program_mode = ProgramMode::INTERACTING;
    } else if (program_mode_str == "training") {
        program_mode = ProgramMode::TRAINING;
    } else if (program_mode_str == "decoding") {
        program_mode = ProgramMode::DECODING;
    } else if (program_mode_str == "metric") {
        program_mode = ProgramMode::METRIC;
    } else {
        cout << format("program mode is %1%") % program_mode_str << endl;
        abort();
    }
    default_config.program_mode = program_mode;

    default_config.check_grad = ini_reader.GetBoolean(SECTION, "check_grad", false);
    default_config.one_response = ini_reader.GetBoolean(SECTION, "one_response", false);
    default_config.learn_test = ini_reader.GetBoolean(SECTION, "learn_test", false);
    default_config.save_model_per_batch = ini_reader.GetBoolean(SECTION, "save_model_per_batch",
            false);
    default_config.split_unknown_words = ini_reader.GetBoolean(SECTION, "split_unknown_words",
            true);

    default_config.max_sample_count = ini_reader.GetInteger(SECTION, "max_sample_count",
            1000000000);
    default_config.hold_batch_size = ini_reader.GetInteger(SECTION, "hold_batch_size", 100);
    default_config.device_id = ini_reader.GetInteger(SECTION, "device_id", 0);
    default_config.seed = ini_reader.GetInteger(SECTION, "seed", 0);
    default_config.cut_length = ini_reader.GetInteger(SECTION, "cut_length", 30);
    default_config.keyword_bound = ini_reader.GetReal(SECTION, "keyword_bound", 0);
    default_config.keyword_fork_bound = ini_reader.GetReal(SECTION, "keyword_fork_bound", 0);
    default_config.output_model_file_prefix = ini_reader.Get(SECTION, "output_model_file_prefix",
            "");
    default_config.input_model_file = ini_reader.Get(SECTION, "input_model_file", "");
    default_config.input_model_dir = ini_reader.Get(SECTION, "input_model_dir", "");
    default_config.black_list_file = ini_reader.Get(SECTION, "black_list_file", "");
    default_config.memory_in_gb = ini_reader.GetReal(SECTION, "memory_in_gb", 0.0f);
    default_config.ngram_penalty_1 = ini_reader.GetReal(SECTION, "ngram_penalty_1", 0.0f);
    default_config.ngram_penalty_2 = ini_reader.GetReal(SECTION, "ngram_penalty_2", 0.0f);
    default_config.ngram_penalty_3 = ini_reader.GetReal(SECTION, "ngram_penalty_3", 0.0f);

    return default_config;
}

HyperParams parseHyperParams(INIReader &ini_reader) {
    HyperParams hyper_params;

    int word_dim = ini_reader.GetInteger("hyper", "word_dim", 0);
    if (word_dim <= 0) {
        cerr << "word_dim wrong" << endl;
        abort();
    }
    hyper_params.word_dim = word_dim;

    int encoding_hidden_dim = ini_reader.GetInteger("hyper", "hidden_dim", 0);
    if (encoding_hidden_dim <= 0) {
        cerr << "hidden_dim wrong" << endl;
        abort();
    }
    hyper_params.hidden_dim = encoding_hidden_dim;

    float dropout = ini_reader.GetReal("hyper", "dropout", 0.0);
    if (dropout < -1.0f || dropout >=1.0f) {
        cerr << "dropout wrong" << endl;
        abort();
    }
    hyper_params.dropout = dropout;

    int batch_size = ini_reader.GetInteger("hyper", "batch_size", 0);
    if (batch_size == 0) {
        cerr << "batch_size not found" << endl;
        abort();
    }
    hyper_params.batch_size = batch_size;

    int beam_size = ini_reader.GetInteger("hyper", "beam_size", 0);
    if (beam_size == 0) {
        cerr << "beam_size not found" << endl;
        abort();
    }
    hyper_params.beam_size = beam_size;

    float learning_rate = ini_reader.GetReal("hyper", "learning_rate", 0.001f);
    if (learning_rate <= 0.0f) {
        cerr << "learning_rate wrong" << endl;
        abort();
    }
    hyper_params.learning_rate = learning_rate;

    float min_learning_rate = ini_reader.GetReal("hyper", "min_learning_rate", 0.0001f);
    if (min_learning_rate <= 0.0f) {
        cerr << "min_learning_rate wrong" << endl;
        abort();
    }
    hyper_params.min_learning_rate = min_learning_rate;

    float learning_rate_decay = ini_reader.GetReal("hyper", "learning_rate_decay", 0.9f);
    if (learning_rate_decay <= 0.0f || learning_rate_decay > 1.0f) {
        cerr << "decay wrong" << endl;
        abort();
    }
    hyper_params.learning_rate_decay = learning_rate_decay;

    float warm_up_learning_rate = ini_reader.GetReal("hyper", "warm_up_learning_rate", 1e-6);
    if (warm_up_learning_rate < 0 || warm_up_learning_rate > 1.0f) {
        cerr << "warm_up_learning_rate wrong" << endl;
        abort();
    }
    hyper_params.warm_up_learning_rate = warm_up_learning_rate;

    int warm_up_iterations = ini_reader.GetInteger("hyper", "warm_up_iterations", 1000);
    if (warm_up_iterations < 0) {
        cerr << "warm_up_iterations wrong" << endl;
        abort();
    }
    hyper_params.warm_up_iterations = warm_up_iterations;

    int word_cutoff = ini_reader.GetReal("hyper", "word_cutoff", -1);
    if(word_cutoff == -1){
   	cerr << "word_cutoff read error" << endl;
    }
    hyper_params.word_cutoff = word_cutoff;

    bool word_finetune = ini_reader.GetBoolean("hyper", "word_finetune", -1);
    hyper_params.word_finetune = word_finetune;

    string word_file = ini_reader.Get("hyper", "word_file", "");
    hyper_params.word_file = word_file;

    float l2_reg = ini_reader.GetReal("hyper", "l2_reg", 0.0f);
    if (l2_reg < 0.0f || l2_reg > 1.0f) {
        cerr << "l2_reg:" << l2_reg << endl;
        abort();
    }
    hyper_params.l2_reg = l2_reg;
    string optimizer = ini_reader.Get("hyper", "optimzer", "");
    if (optimizer == "adam") {
        hyper_params.optimizer = Optimizer::ADAM;
    } else if (optimizer == "adagrad") {
        hyper_params.optimizer = Optimizer::ADAGRAD;
    } else if (optimizer == "adamw") {
        hyper_params.optimizer = Optimizer::ADAMW;
    } else {
        cerr << "invalid optimzer:" << optimizer << endl;
        abort();
    }

    float idf_threshhold = ini_reader.GetReal("hyper", "idf_threshhold", -1.0f);
    if (idf_threshhold < 0) {
        cerr << "idf_threshhold:" << idf_threshhold << endl;
        abort();
    }
    hyper_params.idf_threshhold = idf_threshhold;

    return hyper_params;
}

vector<int> toNormalWordIds(const vector<string> &sentence,
        const LookupTable<Param> &normal_table) {
    vector<int> ids;
    for (const string &word : sentence) {
        const auto &it = normal_table.elems.m_string_to_id.find(word);
        int xid;
        if (it == normal_table.elems.m_string_to_id.end()) {
            xid = normal_table.nVSize;
        } else {
            xid = it->second;
        }
        ids.push_back(xid);
    }
    return ids;
}

vector<int> toKeywordIds(const vector<string> &sentence, const LookupTable<Param> &keyword_table) {
    vector<int> ids;
    for (const string &word : sentence) {
	int xid = keyword_table.getElemId(word);
        if (xid >= keyword_table.nVSize) {
            cerr << "xid:" << xid << " word:" << word << endl;
            for (const string &w :sentence) {
                cerr << w;
            }
            cerr << endl;
            abort();
        }
        ids.push_back(xid);
    }
    return ids;
}

void printWordIds(const vector<int> &word_ids, const LookupTable<Param> &lookup_table) {
    for (int word_id : word_ids) {
        cout << (word_id < lookup_table.nVSize ? lookup_table.elems.from_id(word_id) : "keyword")
            << " ";
    }
    cout << endl;
}

void printWordIdsWithKeywords(const vector<int> &word_ids,
        const LookupTable<Param> &lookup_table) {
    for (int i = 0; i < word_ids.size(); i += 2) {
        int word_id = word_ids.at(i);
        cout << lookup_table.elems.from_id(word_id) << " ";
    }
    cout << endl;
    for (int i = 1; i < word_ids.size(); i += 2) {
        int word_id = word_ids.at(i);
        cout << lookup_table.elems.from_id(word_id) << " ";
    }
    cout << endl;
}

void analyze(const vector<int> &results, const vector<int> &answers, Metric &metric) {
    if (results.size() != answers.size()) {
        cerr << "results:" << endl;
        for (int result : results) {
            cerr << result << " ";
        }
        cerr << endl << "answers:" << endl;
        for (int answer : answers) {
            cerr << answer << " ";
        }
        cerr << endl;
        cerr << "results size is not equal to answers size" << endl;
        cerr << boost::format("results size:%1% answers size:%2%\n") % results.size() %
            answers.size();
        abort();
    }

    int size = results.size();
    for (int i = 0; i < size; ++i) {
        ++metric.overall_label_count;
        if (results.at(i) == answers.at(i)) {
            ++metric.correct_label_count;
        }
    }
}

string saveModel(const HyperParams &hyper_params, ModelParams &model_params,
        const string &filename_prefix, int epoch) {
    cout << "saving model file..." << endl;
    auto t = time(nullptr);
    auto tm = *localtime(&t);
    ostringstream oss;
    oss << put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    string filename = filename_prefix + oss.str() + "-epoch" + to_string(epoch);
#if USE_GPU
    model_params.copyFromDeviceToHost();
#endif

    Json::Value root;
    root["hyper_params"] = hyper_params.toJson();
    root["model_params"] = model_params.toJson();
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "";
    string json_str = Json::writeString(builder, root);
    ofstream out(filename);
    out << json_str;
    out.close();
    cout << format("model file %1% saved") % filename << endl;
    return filename;
}

shared_ptr<Json::Value> loadModel(const string &filename) {
    ifstream is(filename.c_str());
    shared_ptr<Json::Value> root(new Json::Value);
    if (is) {
        cout << "loading model..." << endl;
        stringstream sstr;
        sstr << is.rdbuf();
        string str = sstr.str();
        Json::CharReaderBuilder builder;
        auto reader = unique_ptr<Json::CharReader>(builder.newCharReader());
        string error;
        if (!reader->parse(str.c_str(), str.c_str() + str.size(), root.get(), &error)) {
            cerr << boost::format("parse json error:%1%") % error << endl;
            abort();
        }
        cout << "model loaded" << endl;
    } else {
        cerr << format("failed to open is, error when loading %1%") % filename << endl;
        abort();
    }

    return root;
}

void loadModel(const DefaultConfig &default_config, HyperParams &hyper_params,
        ModelParams &model_params,
        const Json::Value *root,
        const function<void(const DefaultConfig &default_config, const HyperParams &hyper_params,
            ModelParams &model_params, const Alphabet*, const Alphabet*)> &allocate_model_params) {
    hyper_params.fromJson((*root)["hyper_params"]);
    hyper_params.print();
    allocate_model_params(default_config, hyper_params, model_params, nullptr, nullptr);
    model_params.fromJson((*root)["model_params"]);
#if USE_GPU
    model_params.copyFromHostToDevice();
#endif
}

pair<vector<Node *>, vector<int>> keywordNodesAndIds(const DecoderComponents &decoder_components,
        const WordIdfInfo &idf_info,
        const ModelParams &model_params) {
    vector<Node *> keyword_result_nodes = decoder_components.keyword_vector_to_onehots;
    vector<int> keyword_ids = toKeywordIds(idf_info.keywords_behind, model_params.keyword_table);
    vector<Node *> non_null_nodes;
    vector<int> changed_keyword_ids;
    for (int j = 0; j < keyword_result_nodes.size(); ++j) {
        if (keyword_result_nodes.at(j) != nullptr) {
            non_null_nodes.push_back(keyword_result_nodes.at(j));
            changed_keyword_ids.push_back(keyword_ids.at(j));
            if (keyword_ids.at(j) == model_params.keyword_table.elems.from_string(::unknownkey)) {
                cerr << "unkownkey keyword found:" << endl;
                abort();
            }
        }
    }

    return {non_null_nodes, changed_keyword_ids};
}


float metricTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences,
        const vector<WordIdfInfo> &post_idf_info_list,
        const vector<WordIdfInfo> &response_idf_info_list) {
    cout << "metricTestPosts begin" << endl;
    hyper_params.print();
    float rep_perplex(0.0f);
    thread_pool pool(16);
    mutex rep_perplex_mutex;

    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        auto f = [&]() {
            cout << "post:" << endl;
            auto post = post_sentences.at(post_and_responses.post_id);
            print(post);
            const WordIdfInfo &post_idf_info = post_idf_info_list.at(post_and_responses.post_id);
            print(post_idf_info.keywords_behind);

            const vector<int> &response_ids = post_and_responses.response_ids;
            float avg_perplex = 0.0f;
            cout << "response size:" << response_ids.size() << endl;
            for (int response_id : response_ids) {
//                cout << "response:" << endl;
//                auto response = response_sentences.at(response_id);
//                print(response);
                const WordIdfInfo &idf_info = response_idf_info_list.at(response_id);
//                print(idf_info.keywords_behind);
                n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
                profiler.BeginEvent("build computation graph");
                Graph graph;
                GraphBuilder graph_builder;
                graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id),
                        post_idf_info.keywords_behind,
                        hyper_params, model_params, false);
                DecoderComponents decoder_components;
                graph_builder.forwardDecoder(graph, decoder_components,
                        response_sentences.at(response_id), idf_info.keywords_behind, hyper_params,
                        model_params, false);
                profiler.EndEvent();
                graph.compute();
                vector<Node*> nodes = toNodePointers(decoder_components.wordvector_to_onehots);
                vector<int> word_ids = transferVector<int, string>(
                        response_sentences.at(response_id), [&](const string &w) -> int {
                        return model_params.lookup_table.getElemId(w);
                        });
                auto keyword_nodes_and_ids = keywordNodesAndIds(decoder_components, idf_info,
                        model_params);
                for (int i = 0; i < keyword_nodes_and_ids.first.size(); ++i) {
                    nodes.push_back(keyword_nodes_and_ids.first.at(i));
                    word_ids.push_back(keyword_nodes_and_ids.second.at(i));
                }

                float perplex = computePerplex(nodes, word_ids);
                avg_perplex += perplex;
            }
            avg_perplex /= response_ids.size();
            cout << "size:" << response_ids.size() << endl;
            cout << "avg_perplex:" << avg_perplex << endl;
            rep_perplex_mutex.lock();
            rep_perplex += avg_perplex;
            rep_perplex_mutex.unlock();
        };
        post(pool, f);
    }
    pool.join();

    cout << "total avg perplex:" << rep_perplex / post_and_responses_vector.size() << endl;
    return rep_perplex;
}

void decodeTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        DefaultConfig &default_config,
        const unordered_map<string, float> & word_idf_table,
        const vector<WordIdfInfo> &post_idf_info_list,
        const vector<WordIdfInfo> &response_idf_info_list,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences,
        const vector<string> &black_list) {
    cout << "decodeTestPosts begin" << endl;
    hyper_params.print();
    vector<CandidateAndReferences> candidate_and_references_vector;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        cout << "post:" << endl;
        auto post_sentence = post_sentences.at(post_and_responses.post_id);
        print(post_sentence);
        const auto &idf = post_idf_info_list.at(post_and_responses.post_id);
        Graph graph;
        GraphBuilder graph_builder;
        graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id),
                idf.keywords_behind, hyper_params, model_params, false);
        vector<DecoderComponents> decoder_components_vector;
        decoder_components_vector.resize(hyper_params.beam_size);
        auto pair = graph_builder.forwardDecoderUsingBeamSearch(graph, decoder_components_vector,
                word_idf_table, hyper_params.beam_size, hyper_params,
                model_params, default_config, black_list);
        const vector<WordIdAndProbability> &word_ids_and_probability = pair.first;
        cout << "post:" << endl;
        print(post_sentences.at(post_and_responses.post_id));
        cout << "response:" << endl;
        printWordIdsWithKeywords(toWordIds(word_ids_and_probability), model_params.lookup_table);
        dtype probability = pair.second;
        cout << format("probability:%1%") % probability << endl;
        if (word_ids_and_probability.empty()) {
            continue;
        }

        const vector<int> &response_ids = post_and_responses.response_ids;
        vector<vector<string>> str_references =
            transferVector<vector<string>, int>(response_ids,
                    [&](int response_id) -> vector<string> {
                    return response_sentences.at(response_id);
                    });
        auto transfer = [&](const WordIdAndProbability &in) ->string {
            return model_params.lookup_table.elems.from_id(in.word_id);
        };
        auto decoded_words = transferVector<string, WordIdAndProbability>(
                word_ids_and_probability, transfer);
        CandidateAndReferences candidate_and_references(decoded_words, str_references);
        candidate_and_references_vector.push_back(candidate_and_references);

        float bleu_value = computeBleu(candidate_and_references_vector, 4);
        cout << "bleu_value:" << bleu_value << endl;
    }
}

void interact(const DefaultConfig &default_config, const HyperParams &hyper_params,
        ModelParams &model_params,
        unordered_map<string, float> &word_idfs,
        unordered_map<string, int> &word_counts,
        int word_cutoff,
        const vector<string> black_list) {
}

pair<unordered_set<int>, unordered_set<int>> PostAndResponseIds(
        const vector<PostAndResponses> &post_and_responses_vector) {
    unordered_set<int> post_ids, response_ids;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        post_ids.insert(post_and_responses.post_id);
        for (int id : post_and_responses.response_ids) {
            response_ids.insert(id);
        }
    }
    return make_pair(post_ids, response_ids);
}

unordered_set<string> knownWords(const unordered_map<string, int> &word_counts, int word_cutoff) {
    unordered_set<string> word_set;
    for (auto it : word_counts) {
        if (it.second > word_cutoff) {
            word_set.insert(it.first);
        }
    }
    return word_set;
}

unordered_set<string> knownWords(const vector<string> &words) {
    unordered_set<string> word_set;
    for (const string& w : words) {
        word_set.insert(w);
    }
    return word_set;
}

int calculateIdfThreshholdOffset(const unordered_map<string, float> &idf_table,
        const vector<string> &word_list,
        float threshhold) {
    float min = 1e10;
    string min_str;
    for (auto &it : idf_table) {
        if (it.second >= threshhold && min > it.second) {
            min = it.second;
            min_str = it.first;
        }
    }

    if (min_str.empty()) {
        cerr << "min_str still empty" << endl;
        abort();
    }

    const auto &it = std::find(word_list.begin(), word_list.end(), min_str);
    if (it == word_list.end()) {
        abort();
    }
    return it - word_list.begin();
}

unordered_map<string, int> getKeywords(const unordered_map<string, float> &idf_table,
        const unordered_map<string, int> &word_count_table,
        float threshhold) {
    unordered_map<string, int> map_result;
    for (auto &it : word_count_table) {
        if (idf_table.at(it.first) >= threshhold || it.first == STOP_SYMBOL) {
            map_result.insert(it);
        }
    }
    return map_result;
}

unordered_map<string, int> getNormalWords(const unordered_map<string, float> &idf_table,
        const unordered_map<string, int> &word_count_table,
        float threshhold) {
    unordered_map<string, int> map_result;
    for (auto &it : word_count_table) {
        if (idf_table.at(it.first) < threshhold && it.first != STOP_SYMBOL) {
            map_result.insert(it);
        }
    }
    return map_result;
}

std::pair<dtype, std::vector<int>> MaxLogProbabilityLossWithInconsistentDims(
        const std::vector<Node*> &result_nodes,
        const std::vector<int> &ids,
        int batchsize,
        int vocabulary_size) {
    if (ids.size() != result_nodes.size()) {
        cerr << "ids size is not equal to result_nodes'." << endl;
        abort();
    }

    pair<dtype, std::vector<int>> final_result;

    for (int i = 0; i < result_nodes.size(); ++i) {
        if (ids.at(i) >= result_nodes.at(i)->getDim() || ids.at(i) < 0) {
            cerr << boost::format("id:%1% dim:%2%") % ids.at(i) % result_nodes.at(i)->getDim() <<
                endl;
            abort();
        }
        vector<int> id = {ids.at(i)};
        vector<Node *> node = {result_nodes.at(i)};
        auto result = maxLogProbabilityLoss(node, id, 1.0 / batchsize);
        if (result.first < 0) {
            cerr << "loss is less than 0:" << result.first << endl;
            cerr << boost::format("node dim:%1% id:%2%") % node.front()->getDim() % id.front() <<
                endl;
            node.front()->getVal().print();
            abort();
        }
        if (result.second.size() != 1) {
            cerr << "result second size:" << result.second.size() << endl;
            abort();
        }
        final_result.first += result.first;
        int word_id = result.second.front();
        if (word_id < 0 || word_id > vocabulary_size) {
            cerr << "word_id:" << word_id << endl;
            abort();
        }
        final_result.second.push_back(word_id);
    }

    return final_result;
}

int main(int argc, char *argv[]) {
    cout << "dtype size:" << sizeof(dtype) << endl;

    Options options("single-turn-conversation", "single turn conversation");
    options.add_options()
        ("config", "config file name", cxxopts::value<string>());
    auto args = options.parse(argc, argv);

    string configfilename = args["config"].as<string>();
    INIReader ini_reader(configfilename);
    if (ini_reader.ParseError() < 0) {
        cerr << "parse ini failed" << endl;
        abort();
    }

    DefaultConfig &default_config = GetDefaultConfig();
    default_config = parseDefaultConfig(ini_reader);
    cout << "default_config:" << endl;
    default_config.print();

#if USE_GPU
    n3ldg_cuda::InitCuda(default_config.device_id, default_config.memory_in_gb);
#endif

    HyperParams hyper_params = parseHyperParams(ini_reader);
    cout << "hyper_params:" << endl;
    hyper_params.print();
    vector<PostAndResponses> train_post_and_responses = readPostAndResponsesVector(
            default_config.train_pair_file);
    cout << "train_post_and_responses_vector size:" << train_post_and_responses.size() <<
        endl;
    vector<PostAndResponses> dev_post_and_responses = readPostAndResponsesVector(
            default_config.dev_pair_file);
    cout << "dev_post_and_responses_vector size:" << dev_post_and_responses.size() <<
        endl;
    vector<PostAndResponses> test_post_and_responses = readPostAndResponsesVector(
            default_config.test_pair_file);
    cout << "test_post_and_responses_vector size:" << test_post_and_responses.size() <<
        endl;

    vector<ConversationPair> train_conversation_pairs;
    for (const PostAndResponses &post_and_responses : train_post_and_responses) {
        vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (ConversationPair &conversation_pair : conversation_pairs) {
            train_conversation_pairs.push_back(move(conversation_pair));
        }
    }

    vector<vector<string>> post_sentences = readSentences(default_config.post_file);
    vector<vector<string>> response_sentences = readSentences(default_config.response_file);

    cout << "dev set:" << endl;
    for (PostAndResponses &i : dev_post_and_responses) {
        print(post_sentences.at(i.post_id));
    }

    cout << "test set:" << endl;
    for (PostAndResponses &i : test_post_and_responses) {
        print(post_sentences.at(i.post_id));
    }

    shared_ptr<Json::Value> root_ptr;
    unordered_map<string, int> word_counts;
    auto wordStat = [&]() {
        for (const ConversationPair &conversation_pair : train_conversation_pairs) {
            const vector<string> &post_sentence = post_sentences.at(conversation_pair.post_id);
            addWord(word_counts, post_sentence);

            const vector<string> &response_sentence = response_sentences.at(
                    conversation_pair.response_id);
            addWord(word_counts, response_sentence);
        }

        if (hyper_params.word_file != "" && !hyper_params.word_finetune) {
            for (const PostAndResponses &dev : dev_post_and_responses){
                const vector<string>&post_sentence = post_sentences.at(dev.post_id);
                addWord(word_counts, post_sentence);

                for(int i=0; i<dev.response_ids.size(); i++){
                    const vector<string>&resp_sentence = response_sentences.at(
                            dev.response_ids.at(i));
                    addWord(word_counts, resp_sentence);
                }
            }

            for (const PostAndResponses &test : test_post_and_responses){
                const vector<string>&post_sentence = post_sentences.at(test.post_id);
                addWord(word_counts, post_sentence);

                for(int i =0; i<test.response_ids.size(); i++){
                    const vector<string>&resp_sentence =
                        response_sentences.at(test.response_ids.at(i));
                    addWord(word_counts, resp_sentence);
                }
            }
        }
    };
    wordStat();

    vector<vector<string>> all_sentences;
    cout << "merging sentences..." << endl;
    for (auto &s : post_sentences) {
        all_sentences.push_back(s);
    }
    for (auto &s : response_sentences) {
        all_sentences.push_back(s);
    }
    cout << "merged" << endl;
    cout << "calculating idf" << endl;
    auto all_idf = calculateIdf(all_sentences);
    cout << "idf calculated" << endl;
    cout << boost::format("idf:%1%") % hyper_params.idf_threshhold << endl;

    int sum = 0;
    int ii = 0;
    for (auto &s : response_sentences) {
        ++ii;
        bool include = false;
        for (const string &w : s) {
            if (all_idf.at(w) > hyper_params.idf_threshhold) {
                break;
            }
        }
        if (include) {
            ++sum;
        }
    }
    cout << boost::format("%1% sentences contain words of idf %2%") %
        ((float)sum / response_sentences.size()) % hyper_params.idf_threshhold << endl;

    Alphabet keyword_alphabet, normal_alphabet;
    auto keyword_counts = getKeywords(all_idf, word_counts, hyper_params.idf_threshhold);
    keyword_counts.insert(make_pair(unknownkey, 1));
    keyword_alphabet.init(keyword_counts, 0);
    auto normal_counts = getNormalWords(all_idf, word_counts, hyper_params.idf_threshhold);
    normal_counts.insert(make_pair(unknownkey, 1));
    normal_alphabet.init(normal_counts, 0);
    if (keyword_alphabet.size() + normal_alphabet.size() != word_counts.size()) {
        cerr << boost::format("keyword size is %1%, normal is %2%, all is %3%") %
            keyword_alphabet.size() % normal_alphabet.size() % word_counts.size() << endl;
    }

    cout << boost::format("keyword voc size:%1% normal size:%2%") % keyword_alphabet.size() %
        normal_alphabet.size() << endl;

    ModelParams model_params;
    int beam_size = hyper_params.beam_size;

    auto allocate_model_params = [](const DefaultConfig &default_config,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const Alphabet *normal_alphabet,
            const Alphabet *keyword_alphabet) {
        cout << format("allocate word_file:%1%\n") % hyper_params.word_file;
        if (keyword_alphabet != nullptr) {
            if (normal_alphabet == nullptr) {
                cerr << "keyword_alphabet is not null, but normal_alphabet is!" << endl;
                abort();
            }
            if(hyper_params.word_file != "" &&
                    default_config.program_mode == ProgramMode::TRAINING &&
                    default_config.input_model_file == "") {
                model_params.keyword_table.init(*keyword_alphabet, hyper_params.word_file,
                        hyper_params.word_finetune);
                model_params.lookup_table.init(*normal_alphabet, hyper_params.word_file,
                        hyper_params.word_finetune);
            } else {
                model_params.keyword_table.init(*keyword_alphabet, hyper_params.word_file,
                        hyper_params.word_finetune);
                model_params.lookup_table.init(*normal_alphabet, hyper_params.word_dim, true);
            }
        }
        model_params.attention_params.init(hyper_params.hidden_dim, hyper_params.hidden_dim);
        model_params.left_to_right_encoder_params.init(hyper_params.hidden_dim,
                2 * hyper_params.word_dim + hyper_params.hidden_dim);
        model_params.hidden_to_wordvector_params.init(hyper_params.word_dim,
                2 * hyper_params.hidden_dim + 3 * hyper_params.word_dim, false);
        model_params.hidden_to_keyword_params.init(hyper_params.word_dim,
                2 * hyper_params.hidden_dim, false);
    };

    if (default_config.program_mode != ProgramMode::METRIC) {
        if (default_config.input_model_file == "") {
            allocate_model_params(default_config, hyper_params, model_params, &normal_alphabet,
                    &keyword_alphabet);
            cout << "complete allocate" << endl;
        } else {
            root_ptr = loadModel(default_config.input_model_file);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            word_counts = model_params.lookup_table.elems.m_string_to_id;
        }
    } else {
        if (default_config.input_model_file == "") {
            abort();
        } else {
            root_ptr = loadModel(default_config.input_model_file);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            word_counts = model_params.lookup_table.elems.m_string_to_id;
        }
    }
    auto black_list = readBlackList(default_config.black_list_file);

    cout << "reading post idf info ..." << endl;
    vector<WordIdfInfo> post_idf_info_list = readWordIdfInfoList(post_sentences, all_idf,
          model_params.lookup_table.elems.m_string_to_id, hyper_params.idf_threshhold);
    cout << "completed" << endl;
    cout << "reading response idf info ..." << endl;
    vector<WordIdfInfo> response_idf_info_list = readWordIdfInfoList(response_sentences, all_idf,
          model_params.lookup_table.elems.m_string_to_id, hyper_params.idf_threshhold);
    cout << "completed" << endl;

    if (default_config.program_mode == ProgramMode::INTERACTING) {
        hyper_params.beam_size = beam_size;
        interact(default_config, hyper_params, model_params, all_idf, word_counts,
                hyper_params.word_cutoff, black_list);
    } else if (default_config.program_mode == ProgramMode::DECODING) {
        hyper_params.beam_size = beam_size;
        decodeTestPosts(hyper_params, model_params, default_config, all_idf, post_idf_info_list,
                response_idf_info_list, test_post_and_responses, post_sentences,
                response_sentences, black_list);
    } else if (default_config.program_mode == ProgramMode::METRIC) {
        path dir_path(default_config.input_model_dir);
        if (!is_directory(dir_path)) {
            cerr << format("%1% is not dir path") % default_config.input_model_dir << endl;
            abort();
        }

        vector<string> ordered_file_paths;
        for(auto& entry : boost::make_iterator_range(directory_iterator(dir_path), {})) {
            string basic_name = entry.path().filename().string();
            cout << format("basic_name:%1%") % basic_name << endl;
            if (basic_name.find("model") != 0) {
                continue;
            }

            string model_file_path = entry.path().string();
            ordered_file_paths.push_back(model_file_path);
        }
        std::sort(ordered_file_paths.begin(), ordered_file_paths.end(),
                [](const string &a, const string &b)->bool {
                using boost::filesystem::last_write_time;
                return last_write_time(a) < last_write_time(b);
                });

        float max_rep_perplex = 0.0f;
        for(const string &model_file_path : ordered_file_paths) {
            cout << format("model_file_path:%1%") % model_file_path << endl;
            ModelParams model_params;
            shared_ptr<Json::Value> root_ptr = loadModel(model_file_path);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            float rep_perplex = metricTestPosts(hyper_params, model_params, dev_post_and_responses,
                    post_sentences, response_sentences, post_idf_info_list,
                    response_idf_info_list);
            cout << format("model %1% rep_perplex is %2%") % model_file_path % rep_perplex << endl;
            if (max_rep_perplex < rep_perplex) {
                max_rep_perplex = rep_perplex;
                cout << format("best model now is %1%, and rep_perplex is %2%") % model_file_path %
                    rep_perplex << endl;
            }
        }
    } else if (default_config.program_mode == ProgramMode::TRAINING) {
        ModelUpdate model_update;
        model_update._alpha = hyper_params.learning_rate;
        model_update._reg = hyper_params.l2_reg;
        model_update.setParams(model_params.tunableParams());

        CheckGrad grad_checker;
        if (default_config.check_grad) {
            grad_checker.init(model_params.tunableParams());
        }

        dtype last_loss_sum = 1e10f;
        dtype loss_sum = 0.0f;

        int iteration = 0;
        string last_saved_model;
        default_random_engine engine(default_config.seed);

        for (int epoch = 0; ; ++epoch) {
            cout << "epoch:" << epoch << endl;

            auto cmp = [&] (const ConversationPair &a, const ConversationPair &b)->bool {
                auto len = [&] (const ConversationPair &pair)->int {
                    return post_sentences.at(pair.post_id).size() +
                        response_sentences.at(pair.response_id).size();
                };
                return len(a) < len(b);
            };
            sort(begin(train_conversation_pairs), end(train_conversation_pairs), cmp);
            int valid_len = train_conversation_pairs.size() / hyper_params.batch_size *
                hyper_params.batch_size;
            int batch_count = valid_len / hyper_params.batch_size;
            cout << boost::format("valid_len:%1% batch_count:%2%") % valid_len % batch_count <<
                endl;
            for (int i = 0; i < hyper_params.batch_size; ++i) {
                auto begin_pos = begin(train_conversation_pairs) + i * batch_count;
                shuffle(begin_pos, begin_pos + batch_count, engine);
            }

            unique_ptr<Metric> metric = unique_ptr<Metric>(new Metric);
            unique_ptr<Metric> keyword_metric = unique_ptr<Metric>(new Metric);
            n3ldg_cuda::Profiler::Reset();
            n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
            profiler.SetEnabled(false);
            profiler.BeginEvent("total");

            for (int batch_i = 0; batch_i < batch_count; ++batch_i) {
                cout << format("batch_i:%1% iteration:%2%") % batch_i % iteration << endl;
                if (epoch == 0) {
                    if (iteration < hyper_params.warm_up_iterations) {
                        model_update._alpha = hyper_params.warm_up_learning_rate;
                    } else {
                        model_update._alpha = hyper_params.learning_rate;
                        cout << "warm up finished, learning rate now:" <<
                            hyper_params.learning_rate << endl;
                    }
                }
                Graph graph;
                vector<shared_ptr<GraphBuilder>> graph_builders;
                vector<DecoderComponents> decoder_components_vector;
                vector<ConversationPair> conversation_pair_in_batch;
                auto getSentenceIndex = [batch_i, batch_count](int i) {
                    return i * batch_count + batch_i;
                };
                for (int i = 0; i < hyper_params.batch_size; ++i) {
                    shared_ptr<GraphBuilder> graph_builder(new GraphBuilder);
                    graph_builders.push_back(graph_builder);
                    int instance_index = getSentenceIndex(i);
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    conversation_pair_in_batch.push_back(train_conversation_pairs.at(
                                instance_index));
                    auto post_sentence = post_sentences.at(post_id);
                    const WordIdfInfo& post_idf = post_idf_info_list.at(post_id);
                    graph_builder->forward(graph, post_sentence, post_idf.keywords_behind,
                            hyper_params, model_params, true);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    auto response_sentence = response_sentences.at(response_id);
                    const WordIdfInfo &idf_info = response_idf_info_list.at(response_id);
                    DecoderComponents decoder_components;
                    graph_builder->forwardDecoder(graph, decoder_components, response_sentence,
                            idf_info.keywords_behind, hyper_params, model_params, true);
                    decoder_components_vector.push_back(decoder_components);
                }

                graph.compute();

                for (int i = 0; i < hyper_params.batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    auto response_sentence = response_sentences.at(response_id);
                    const WordIdfInfo &response_idf = response_idf_info_list.at(response_id);
                    vector<Node*> result_nodes =
                        toNodePointers(decoder_components_vector.at(i).wordvector_to_onehots);
                    profiler.BeginEvent("loss");

                    vector<int> normal_ids = toNormalWordIds(response_sentence,
                            model_params.lookup_table);
                    auto result = MaxLogProbabilityLossWithInconsistentDims(result_nodes,
                            normal_ids, hyper_params.batch_size, model_params.lookup_table.nVSize);
                    profiler.EndCudaEvent();
                    loss_sum += result.first;
                    analyze(result.second, normal_ids, *metric);
                    auto keyword_nodes_and_ids = keywordNodesAndIds(
                            decoder_components_vector.at(i), response_idf, model_params);
                    vector<Node *> keyword_nodes = keyword_nodes_and_ids.first;

                    profiler.BeginEvent("loss");
                    auto keyword_result = MaxLogProbabilityLossWithInconsistentDims(
                            keyword_nodes_and_ids.first, keyword_nodes_and_ids.second,
                            hyper_params.batch_size, model_params.keyword_table.nVSize);
                    profiler.EndCudaEvent();
                    if (keyword_result.first < 0) {
                        cerr << boost::format("keyword result is less than 0:%1%") %
                            keyword_result.first << endl;
                        abort();
                    }
                    loss_sum += keyword_result.first;
                    analyze(keyword_result.second, keyword_nodes_and_ids.second, *keyword_metric);

                    static int count_for_print;
                    if (++count_for_print % 100 == 1) {
                        int post_id = train_conversation_pairs.at(instance_index).post_id;
                        cout << "post:" << post_id << endl;
                        print(post_sentences.at(post_id));

                        cout << "golden answer:" << endl;
                        printWordIds(normal_ids, model_params.lookup_table);
                        cout << "output:" << endl;
                        printWordIds(result.second, model_params.lookup_table);

                        cout << "golden keywords:" << endl;
                        printWordIds(keyword_nodes_and_ids.second, model_params.keyword_table);
                        cout << "output:" << endl;
                        printWordIds(keyword_result.second, model_params.keyword_table);
                    }
                }

                cout << "loss:" << loss_sum << endl;
                cout << "normal:" << endl;
                metric->print();
                cout << "keyword:" << endl;
                keyword_metric->print();

                graph.backward();

                if (default_config.check_grad) {
                    auto loss_function = [&](const ConversationPair &conversation_pair) -> dtype {
                        GraphBuilder graph_builder;
                        Graph graph;

                        graph_builder.forward(graph, post_sentences.at(conversation_pair.post_id),
                                post_idf_info_list.at(conversation_pair.post_id).keywords_behind,
                                hyper_params, model_params, true);

                        DecoderComponents decoder_components;
                        graph_builder.forwardDecoder(graph, decoder_components,
                                response_sentences.at(conversation_pair.response_id),
                                response_idf_info_list.at(
                                    conversation_pair.response_id).keywords_behind, hyper_params,
                                model_params, true);

                        graph.compute();

                        const WordIdfInfo &response_idf = response_idf_info_list.at(
                                conversation_pair.response_id);
                        vector<int> word_ids = toNormalWordIds(
                                response_sentences.at(conversation_pair.response_id),
                                model_params.lookup_table);
                        vector<Node*> result_nodes = toNodePointers(
                                decoder_components.wordvector_to_onehots);
                        auto keyword_nodes_and_ids = keywordNodesAndIds(decoder_components,
                                response_idf, model_params);
                        vector<int> raw_keyword_ids = keyword_nodes_and_ids.second;
                        vector<Node *> keyword_nodes = keyword_nodes_and_ids.first;
                        return MaxLogProbabilityLossWithInconsistentDims(
                                keyword_nodes_and_ids.first, raw_keyword_ids, 1,
                                model_params.keyword_table.nVSize).first +
                            MaxLogProbabilityLossWithInconsistentDims(result_nodes, word_ids,
                                    1, model_params.lookup_table.nVSize).first;
                    };
                    cout << format("checking grad - conversation_pair size:%1%") %
                        conversation_pair_in_batch.size() << endl;
                    grad_checker.check<ConversationPair>(loss_function, conversation_pair_in_batch,
                            "");
                }

                if (hyper_params.optimizer == Optimizer::ADAM) {
                    model_update.updateAdam(10.0f);
                } else if (hyper_params.optimizer == Optimizer::ADAGRAD) {
                    model_update.update(10.0f);
                } else if (hyper_params.optimizer == Optimizer::ADAMW) {
                    model_update.updateAdamW(10.0f);
                } else {
                    cerr << "no optimzer set" << endl;
                    abort();
                }

                if (default_config.save_model_per_batch) {
                    saveModel(hyper_params, model_params, default_config.output_model_file_prefix,
                            epoch);
                }

                ++iteration;
            }
            profiler.EndCudaEvent();
            profiler.Print();

            cout << "loss_sum:" << loss_sum << " last_loss_sum:" << last_loss_sum << endl;
            if (loss_sum > last_loss_sum) {
                if (epoch == 0) {
                    cerr << "loss is larger than last epoch but epoch is 0" << endl;
                    abort();
                }
                model_update._alpha *= 0.1f;
                hyper_params.learning_rate = model_update._alpha;
                cout << "learning_rate decay:" << model_update._alpha << endl;
                std::shared_ptr<Json::Value> root = loadModel(last_saved_model);
                model_params.fromJson((*root)["model_params"]);
#if USE_GPU
                model_params.copyFromHostToDevice();
#endif
            } else {
                model_update._alpha = (model_update._alpha - hyper_params.min_learning_rate) *
                    hyper_params.learning_rate_decay + hyper_params.min_learning_rate;
                hyper_params.learning_rate = model_update._alpha;
                cout << "learning_rate now:" << hyper_params.learning_rate << endl;
                last_saved_model = saveModel(hyper_params, model_params,
                        default_config.output_model_file_prefix, epoch);
            }

            last_loss_sum = loss_sum;
            loss_sum = 0;
        }
    } else {
        abort();
    }

    return 0;
}
