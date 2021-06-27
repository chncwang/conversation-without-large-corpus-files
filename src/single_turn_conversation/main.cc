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
#include <array>
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
#include "insnet/insnet.h"
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
        cerr << "pair file empty" << endl;
        abort();
    }

    default_config.dev_pair_file = ini_reader.Get(SECTION, "dev_pair_file", "");
    if (default_config.dev_pair_file.empty()) {
        cerr << "pair file empty" << endl;
        abort();
    }

    default_config.test_pair_file = ini_reader.Get(SECTION, "test_pair_file", "");
    if (default_config.test_pair_file.empty()) {
        cerr << "pair file empty" << endl;
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

    default_config.decoded_file = ini_reader.Get(SECTION, "decoded_file", "");

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
    } else if (program_mode_str == "decoded_ppl") {
        program_mode = ProgramMode::DECODED_PPL;
    }
    else {
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

    default_config.train_sample_count = ini_reader.GetInteger(SECTION, "train_sample_count",
            1000000000);
    default_config.dev_sample_count = ini_reader.GetInteger(SECTION, "dev_sample_count",
            1000000000);
    default_config.test_sample_count = ini_reader.GetInteger(SECTION, "test_sample_count",
            1000000000);
    default_config.hold_batch_size = ini_reader.GetInteger(SECTION, "hold_batch_size", 100);
    default_config.device_id = ini_reader.GetInteger(SECTION, "device_id", 0);
    default_config.seed = ini_reader.GetInteger(SECTION, "seed", 0);
    default_config.cut_length = ini_reader.GetInteger(SECTION, "cut_length", 30);
    default_config.max_epoch = ini_reader.GetInteger(SECTION, "max_epoch", 100);
    default_config.output_model_file_prefix = ini_reader.Get(SECTION, "output_model_file_prefix",
            "");
    default_config.input_model_file = ini_reader.Get(SECTION, "input_model_file", "");
    default_config.input_model_dir = ini_reader.Get(SECTION, "input_model_dir", "");
    default_config.black_list_file = ini_reader.Get(SECTION, "black_list_file", "");
    default_config.memory_in_gb = ini_reader.GetReal(SECTION, "memory_in_gb", 0.0f);
    default_config.ngram_penalty_1 = ini_reader.GetReal(SECTION, "ngram_penalty_1", 0.0f);
    default_config.ngram_penalty_2 = ini_reader.GetReal(SECTION, "ngram_penalty_2", 0.0f);
    default_config.ngram_penalty_3 = ini_reader.GetReal(SECTION, "ngram_penalty_3", 0.0f);
    default_config.result_count_factor = ini_reader.GetReal(SECTION, "result_count_factor", 1.0f);

    return default_config;
}

HyperParams parseHyperParams(INIReader &ini_reader) {
    HyperParams hyper_params;

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

    hyper_params.clip_grad = ini_reader.GetReal("hyper", "clip_grad", 10.0);

    float learning_rate = ini_reader.GetReal("hyper", "learning_rate", 0.001f);
    if (learning_rate <= 0.0f) {
        cerr << "learning_rate wrong" << endl;
        abort();
    }
    hyper_params.learning_rate = learning_rate;

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

    string optimizer = ini_reader.Get("hyper", "optimzer", "");
    if (optimizer == "adam") {
        hyper_params.optimizer = ::Optimizer::ADAM;
    } else if (optimizer == "adagrad") {
        hyper_params.optimizer = ::Optimizer::ADAGRAD;
    } else if (optimizer == "adamw") {
        hyper_params.optimizer = ::Optimizer::ADAMW;
    } else {
        cerr << "invalid optimzer:" << optimizer << endl;
        abort();
    }

    return hyper_params;
}

vector<int> toIds(const vector<string> &sentence, Embedding<Param> &lookup_table) {
    vector<int> ids;
    for (const string &word : sentence) {
	int xid = lookup_table.getElemId(word);
        ids.push_back(xid);
    }
    return ids;
}

void analyze(const vector<int> &results, const vector<int> &answers, Metric &metric) {
    if (results.size() != answers.size()) {
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

    ofstream out(filename, ios::binary);
    cereal::BinaryOutputArchive output_ar(out);
    output_ar(hyper_params, model_params, epoch);
    cout << format("model file %1% saved") % filename << endl;
    return filename;
}

void loadModel(const DefaultConfig &default_config, HyperParams &hyper_params,
        ModelParams &model_params,
        int &epoch,
        const string &filename,
        const function<void(const DefaultConfig &default_config, const HyperParams &hyper_params,
            ModelParams &model_params, const Vocab*)> &allocate_model_params) {
    ifstream is(filename.c_str());
    if (is) {
        cout << "loading model..." << endl;
        cereal::BinaryInputArchive ar(is);
        ar(hyper_params);
        hyper_params.print();
        allocate_model_params(default_config, hyper_params, model_params, nullptr);
        ar(model_params, epoch);
#if USE_GPU
        model_params.copyFromHostToDevice();
#endif
        cout << "model loaded" << endl;
    } else {
        cerr << fmt::format("load model fail - filename:%1%", filename) << endl;
        abort();
    }
}

float metricTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences) {
    cout << "metricTestPosts begin" << endl;
    hyper_params.print();
    float perplex(0.0f), corpus_hit_sum(0);
    vector<int> corpus_pos_hit_amount, corpus_pos_amount;
    int size_sum = 0;

    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
            cout << "post:" << endl;
            print(post_sentences.at(post_and_responses.post_id));

            const vector<int> &response_ids = post_and_responses.response_ids;
            float sum = 0.0f;
            int hit_sum = 0;
            int word_sum = 0;
            vector<int> post_hit_counts, post_pos_amounts;
            cout << "response size:" << response_ids.size() << endl;
            for (int response_id : response_ids) {
                //            cout << "response:" << endl;
                //            print(response_sentences.at(response_id));
                Graph graph(ModelStage::INFERENCE);
                GraphBuilder graph_builder;
                graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id),
                        hyper_params, model_params);
                Node *node = graph_builder.forwardDecoder(response_sentences.at(response_id),
                        hyper_params, model_params);
                graph.forward();
                vector<int> word_ids = transferVector<int, string>(
                        response_sentences.at(response_id), [&](const string &w) -> int {
                        return model_params.lookup_table.getElemId(w);
                        });
                int hit_count;
                vector<int> hit_flags;
                float perplex = computePerplex(*node, model_params.lookup_table.vocab.size(),
                        word_ids, hit_count, hit_flags, 5);
                sum += perplex;
                hit_sum += hit_count;
                word_sum += word_ids.size();
                for (int i = 0; i < hit_flags.size(); ++i) {
                    if (post_hit_counts.size() <= i) {
                        post_hit_counts.push_back(0);
                    }
                    post_hit_counts.at(i) += hit_flags.at(i);

                    if (post_pos_amounts.size() <= i) {
                        post_pos_amounts.push_back(0);
                    }
                    ++ post_pos_amounts.at(i);
                }
            }
            cout << "avg_perplex:" << exp(sum/word_sum) << endl;
            cout << "avg_hit:" << static_cast<float>(hit_sum) / word_sum << endl;
            for (int i = 0; i < post_hit_counts.size(); ++i) {
                cout << i << " " << static_cast<float>(post_hit_counts.at(i)) /
                        post_pos_amounts.at(i) << endl;
            }
            perplex += sum;
            corpus_hit_sum += hit_sum;
            size_sum += word_sum;

            for (int i = 0; i < post_hit_counts.size(); ++i) {
                if (corpus_pos_amount.size() <= i) {
                    corpus_pos_amount.push_back(0);
                }
                corpus_pos_amount.at(i) += post_pos_amounts.at(i);
                if (corpus_pos_hit_amount.size() <= i) {
                    corpus_pos_hit_amount.push_back(0);
                }
                corpus_pos_hit_amount.at(i) += post_hit_counts.at(i);
            }
    }

    perplex = exp(perplex / size_sum);
    cout << "total avg perplex:" << perplex << endl;
    cout << "corpus hit rate:" << static_cast<float>(corpus_hit_sum) / size_sum << endl;
    cout << "corpus_pos_hit_amount size:" << corpus_pos_hit_amount.size() << endl;
    for (int i = 0; i < corpus_pos_hit_amount.size(); ++i) {
        cout << boost::format("pos:%1% hit:%2% all:%3% rate:%4%") % i %
            corpus_pos_hit_amount.at(i) % corpus_pos_amount.at(i) %
            (static_cast<float>(corpus_pos_hit_amount.at(i)) / corpus_pos_amount.at(i)) << endl;
    }
    return perplex;
}

void computeMeanAndStandardDeviation(const vector<float> &nums, float &mean, float &sd) {
    float sum = 0;
    for (float num : nums) {
        sum += num;
    }
    mean = sum / nums.size();
    if (nums.size() == 1) {
        sd = 0;
    } else {
        float variance = 0;
        for (float num : nums) {
            float x = num - mean;
            variance += x * x;
        }
        variance /= (nums.size() - 1);
        sd = sqrt(variance);
    }
}

//void decodeTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
//        DefaultConfig &default_config,
//        const vector<PostAndResponses> &post_and_responses_vector,
//        const vector<vector<string>> &post_sentences,
//        const vector<vector<string>> &response_sentences,
//        const unordered_map<string, float> &all_idf,
//        const vector<string> &black_list) {
//    LookupTable<Param> original_embeddings;
//    original_embeddings.init(model_params.decoder_lookup_table.elems, hyper_params.word_file);

//    vector<vector<vector<string>>> ref_sentences;
//    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
//        vector<vector<string>> pair;
//        for (const int id : post_and_responses.response_ids) {
//            auto s = response_sentences.at(id);
//            s.pop_back();
//            pair.push_back(s);
//        }
//        ref_sentences.push_back(pair);
//    }
//    using std::array;
//    array<unordered_map<string, float>, 4> ngram_idf_tables;
//    for (int i = 1; i <= 4; ++i) {
//        ngram_idf_tables.at(i - 1) = computeNgramIdf(ref_sentences, i);
//        for (const auto &it : ngram_idf_tables.at(i - 1)) {
//            cout << it.first << ":" << it.second << endl;
//        }
//    }

//    cout << "decodeTestPosts begin" << endl;
//    hyper_params.print();
//    vector<CandidateAndReferences> candidate_and_references_vector;
//    map<string, int64_t> overall_flops;
//    array<float, 4> cider_sums = {0, 0, 0, 0};
//    int64_t activations_sum = 0;
//    int loop_i = 0;
//    vector <float> greedy_matching_similarities;
//    vector <float> avg_matching_similarities;
//    vector <float> extrema_similarities;
//    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
//        ++loop_i;
//        cout << "post:" << endl;
//        print(post_sentences.at(post_and_responses.post_id));
//        Graph graph(false, true, true);
//        GraphBuilder graph_builder;
//        graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id), hyper_params,
//                model_params, false);
//        vector<DecoderComponents> decoder_components_vector;
//        decoder_components_vector.resize(hyper_params.beam_size);
//        auto pair = graph_builder.forwardDecoderUsingBeamSearch(graph, decoder_components_vector,
//                hyper_params.beam_size, hyper_params, model_params, default_config, black_list);
//        const vector<WordIdAndProbability> &word_ids_and_probability = pair.first;
//        cout << "post:" << endl;
//        print(post_sentences.at(post_and_responses.post_id));
//        cout << "response:" << endl;
//        printWordIds(word_ids_and_probability, model_params.decoder_lookup_table);
//        cout << "response words:" << endl;
//        printWordIds(word_ids_and_probability, model_params.decoder_lookup_table, true);
//        const auto &flops = graph.getFLOPs();
//        if (loop_i == 1) {
//            overall_flops = flops;
//        } else {
//            for (const auto &it : flops) {
//                overall_flops.at(it.first) += it.second;
//            }
//        }

//        float flops_sum = 0;
//        for (const auto &it : overall_flops) {
//            flops_sum += it.second;
//        }
//        for (const auto &it : overall_flops) {
//            cout << it.first << ":" << it.second / flops_sum << endl;
//        }

//        cout << boost::format("flops overall:%1% avg:%2%") % flops_sum %
//            (static_cast<float>(flops_sum) / loop_i) << endl;

//        activations_sum += graph.getActivations();
//        cout << boost::format("activations:%1% avg:%2%") % activations_sum %
//            (static_cast<float>(activations_sum) / loop_i) << endl;

//        dtype probability = pair.second;
//        cout << format("probability:%1%") % probability << endl;
//        if (word_ids_and_probability.empty()) {
//            cerr << "empty result" << endl;
//            abort();
//        }

//        vector<string> decoded_word_ids;
//        auto to_str = [&](const WordIdAndProbability &in) ->string {
//            return model_params.decoder_lookup_table.elems.from_id(in.word_id);
//        };
//        transform(word_ids_and_probability.begin(), word_ids_and_probability.end(),
//                back_inserter(decoded_word_ids), to_str);
//        decoded_word_ids.pop_back();
//        const vector<int> &response_ids = post_and_responses.response_ids;
//        vector<vector<string>> str_references =
//            transferVector<vector<string>, int>(response_ids,
//                    [&](int response_id) -> vector<string> {
//                    return response_sentences.at(response_id);
//                    });
//        vector<vector<string>> id_references;
//        for (const vector<string> &strs : str_references) {
//            auto stop_removed = strs;
//            stop_removed.pop_back();
//            id_references.push_back(stop_removed);
//        }

//        CandidateAndReferences candidate_and_references(decoded_word_ids, id_references);
//        candidate_and_references_vector.push_back(candidate_and_references);

//        for (int ngram = 1; ngram <=4; ++ngram) {
//            float bleu_value = computeMtevalBleu(candidate_and_references_vector, ngram);
//            cout << "bleu_" << ngram << ":" << bleu_value << endl;
//            float bleu_mean, bleu_deviation;
//            computeMtevalBleuForEachResponse(candidate_and_references_vector, ngram, bleu_mean,
//                    bleu_deviation);
//            cout << boost::format("bleu_%1% mean:%2% deviation:%3%") % ngram % bleu_mean %
//                bleu_deviation << endl;
//            float nist_value = computeNist(candidate_and_references_vector, ngram);
//            cout << "nist_" << ngram << ":" << nist_value << endl;
//            float dist_value = computeDist(candidate_and_references_vector, ngram);
//            cout << "dist_" << ngram << ":" << dist_value << endl;
//            float cider = computeCIDEr(candidate_and_references, ngram_idf_tables.at(ngram - 1),
//                    ngram);
//            cider_sums.at(ngram - 1) += cider;
//            cout << "cider_" << ngram << ":" << cider_sums.at(ngram - 1) / loop_i << endl;
//        }
//        float idf_value = computeEntropy(candidate_and_references_vector, all_idf);
//        cout << "idf:" << idf_value << endl;
//        float matched_idf = computeMatchedEntropy(candidate_and_references_vector, all_idf);
//        cout << "matched idf:" << matched_idf << endl;
//        float greedy_matching_sim = computeGreedyMatching(candidate_and_references,
//                original_embeddings);
//        greedy_matching_similarities.push_back(greedy_matching_sim);
//        float greedy_matching_sim_mean, greedy_matching_sim_sd;
//        computeMeanAndStandardDeviation(greedy_matching_similarities, greedy_matching_sim_mean,
//                greedy_matching_sim_sd);
//        cout << boost::format("greedy matching mean:%1% standard_deviation:%2%") %
//            greedy_matching_sim_mean % greedy_matching_sim_sd << endl;
//        float avg_sim = computeEmbeddingAvg(candidate_and_references, original_embeddings);
//        avg_matching_similarities.push_back(avg_sim);
//        float avg_matching_sim_mean, avg_matching_sim_sd;
//        computeMeanAndStandardDeviation(avg_matching_similarities, avg_matching_sim_mean,
//                avg_matching_sim_sd);
//        cout << boost::format("embedding average mean:%1% standard_deviation:%2%") %
//            avg_matching_sim_mean % avg_matching_sim_sd << endl;
//        float extrema = computeExtrema(candidate_and_references, original_embeddings);
//        extrema_similarities.push_back(extrema);
//        float extrema_mean, extrema_sd;
//        computeMeanAndStandardDeviation(extrema_similarities, extrema_mean, extrema_sd);
//        cout << boost::format("extrema mean:%1% standard_deviation:%2%") % extrema_mean %
//            extrema_sd << endl;
//    }
//}

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

template<typename T>
void preserveVector(vector<T> &vec, int count, int seed) {
    default_random_engine engine(seed);
    shuffle(vec.begin(), vec.end(), engine);
    vec.erase(vec.begin() + std::min<int>(count, vec.size()), vec.end());
}

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

        utf8_string utf8(it.first);
//        if (includePunctuation(utf8.cpp_str()) || !isPureChinese(it.first) || idf <= 5) {
//            idf = -idf;
//        }

        result.insert(make_pair(it.first, idf));
    }

    return result;
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
    globalPoolEnabled() = (default_config.program_mode == ProgramMode::TRAINING);

#if USE_GPU
    cuda::initCuda(default_config.device_id, default_config.memory_in_gb);
#endif

    HyperParams hyper_params = parseHyperParams(ini_reader);
    cout << "hyper_params:" << endl;
    hyper_params.print();

    vector<PostAndResponses> train_post_and_responses = readPostAndResponsesVector(
            default_config.train_pair_file);
    preserveVector(train_post_and_responses, default_config.train_sample_count,
            default_config.seed);
    cout << "train_post_and_responses_vector size:" << train_post_and_responses.size()
        << endl;
    vector<PostAndResponses> dev_post_and_responses = readPostAndResponsesVector(
            default_config.dev_pair_file);
    preserveVector(dev_post_and_responses, default_config.dev_sample_count, default_config.seed);
    cout << "dev_post_and_responses_vector size:" << dev_post_and_responses.size()
        << endl;
    vector<PostAndResponses> test_post_and_responses = readPostAndResponsesVector(
            default_config.test_pair_file);
    preserveVector(test_post_and_responses, default_config.test_sample_count, default_config.seed);
    cout << "test_post_and_responses_vector size:" << test_post_and_responses.size()
        << endl;
    vector<ConversationPair> train_conversation_pairs;
    for (const PostAndResponses &post_and_responses : train_post_and_responses) {
        vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (ConversationPair &conversation_pair : conversation_pairs) {
            train_conversation_pairs.push_back(move(conversation_pair));
        }
    }

    cout << "train size:" << train_conversation_pairs.size() << " dev size:" <<
        dev_post_and_responses.size() << " test size:" << test_post_and_responses.size() << endl;

    vector<vector<string>> post_sentences = readSentences(default_config.post_file);
    vector<vector<string>> response_sentences = readSentences(default_config.response_file);

    vector<vector<string>> all_sentences;
    for (auto &p : train_conversation_pairs) {
        auto &s = response_sentences.at(p.response_id);
        all_sentences.push_back(s);
        auto &s2 = post_sentences.at(p.post_id);
        all_sentences.push_back(s2);
    }

    auto all_idf = calculateIdf(all_sentences);

    Vocab alphabet;
    unordered_map<string, int> word_counts;
    if (default_config.program_mode == ProgramMode::TRAINING) {
        auto wordStat = [&]() {
            for (const ConversationPair &conversation_pair : train_conversation_pairs) {
                const vector<string> &post_sentence = post_sentences.at(conversation_pair.post_id);
                addWord(word_counts, post_sentence);

                const vector<string> &response_sentence = response_sentences.at(
                        conversation_pair.response_id);
                addWord(word_counts, response_sentence);
            }
        };
        wordStat();
        word_counts[insnet::UNKNOWN_WORD] = 1000000000;
        alphabet.init(word_counts, hyper_params.word_cutoff);
        cout << boost::format("post alphabet size:%1%") % alphabet.size() << endl;
    }

    ModelParams model_params;
    int beam_size = 10;

    auto allocate_model_params = [](const DefaultConfig &default_config,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const Vocab *alphabet) {
        if (alphabet != nullptr) {
            model_params.lookup_table.init(*alphabet, hyper_params.hidden_dim, true);
        }
        model_params.attention_params.init(hyper_params.hidden_dim * 2, hyper_params.hidden_dim);
        model_params.l2r_encoder_params.init(hyper_params.hidden_dim, hyper_params.hidden_dim);
        model_params.r2l_encoder_params.init(hyper_params.hidden_dim, hyper_params.hidden_dim);
        model_params.decoder_params.init(hyper_params.hidden_dim, 3 * hyper_params.hidden_dim);
    };

    int saved_epoch = -1;
    if (default_config.program_mode != ProgramMode::METRIC) {
        if (default_config.input_model_file == "") {
            allocate_model_params(default_config, hyper_params, model_params, &alphabet);
        } else {
            loadModel(default_config, hyper_params, model_params, saved_epoch,
                    default_config.input_model_file,allocate_model_params);
        }
    }

    auto black_list = readBlackList(default_config.black_list_file);

    if (default_config.program_mode == ProgramMode::DECODING) {
//        hyper_params.beam_size = beam_size;
//        decodeTestPosts(hyper_params, model_params, default_config, test_post_and_responses,
//                post_sentences, response_sentences, all_idf, black_list);
    } else if (default_config.program_mode == ProgramMode::DECODED_PPL) {
    } else if (default_config.program_mode == ProgramMode::METRIC) {
//        path dir_path(default_config.input_model_dir);
//        if (!is_directory(dir_path)) {
//            cerr << format("%1% is not dir path") % default_config.input_model_dir << endl;
//            abort();
//        }

//        vector<string> ordered_file_paths;
//        for(auto& entry : boost::make_iterator_range(directory_iterator(dir_path), {})) {
//            string basic_name = entry.path().filename().string();
//            cout << format("basic_name:%1%") % basic_name << endl;
//            if (basic_name.find("model") != 0) {
//                continue;
//            }

//            string model_file_path = entry.path().string();
//            ordered_file_paths.push_back(model_file_path);
//        }
//        std::sort(ordered_file_paths.begin(), ordered_file_paths.end(),
//                [](const string &a, const string &b)->bool {
//                using boost::filesystem::last_write_time;
//                return last_write_time(a) < last_write_time(b);
//                });

//        float min_perplex = 0.0f;
//        for(const string &model_file_path : ordered_file_paths) {
//            cout << format("model_file_path:%1%") % model_file_path << endl;
//            ModelParams model_params;
//            shared_ptr<Json::Value> root_ptr = loadModel(model_file_path);
//            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
//                    allocate_model_params);
//            float perplex = metricTestPosts(hyper_params, model_params,
//                    dev_post_and_responses, post_sentences, response_sentences);
//            cout << format("model %1% perplex is %2%") % model_file_path % perplex << endl;
//            if (min_perplex > perplex) {
//                min_perplex = perplex;
//                cout << format("best model now is %1%, and perplex is %2%") % model_file_path %
//                    perplex << endl;
//            }
//        }
    } else if (default_config.program_mode == ProgramMode::TRAINING) {
        insnet::AdamOptimzer optimizer(model_params.tunableParams());

        CheckGrad grad_checker;
        if (default_config.check_grad) {
            grad_checker.init(model_params.tunableParams());
        }

        dtype loss_sum = 0.0f;

        insnet::Profiler &profiler = insnet::Profiler::Ins();
        profiler.SetEnabled(false);
        profiler.BeginEvent("total");

        string last_saved_model;
        int corpus_word_sum = 0;

        for (int epoch = saved_epoch + 1; epoch < default_config.max_epoch; ++epoch) {
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
            default_random_engine engine(default_config.seed);
            for (int i = 0; i < hyper_params.batch_size; ++i) {
                auto begin_pos = begin(train_conversation_pairs) + i * batch_count;
                shuffle(begin_pos, begin_pos + batch_count, engine);
            }
            if (train_conversation_pairs.size() > hyper_params.batch_size * batch_count) {
                shuffle(begin(train_conversation_pairs) + hyper_params.batch_size * batch_count,
                        train_conversation_pairs.end(), engine);
            }


            unique_ptr<Metric> metric = unique_ptr<Metric>(new Metric);
            using namespace std::chrono;
            int duration_count = 1e3;

            int iteration = epoch * batch_count;

            for (int batch_i = 0; batch_i < batch_count +
                    (train_conversation_pairs.size() > hyper_params.batch_size * batch_count);
                    ++batch_i) {
                auto start = high_resolution_clock::now();
                if (batch_i % 10 == 5)
                    cout << format("batch_i:%1% iteration:%2%") % batch_i % iteration << endl;
                int batch_size = batch_i == batch_count ?
                    train_conversation_pairs.size() % hyper_params.batch_size :
                    hyper_params.batch_size;
                profiler.BeginEvent("build braph");
                Graph graph;
                vector<shared_ptr<GraphBuilder>> graph_builders;
                vector<ConversationPair> conversation_pair_in_batch;
                auto getSentenceIndex = [batch_i, batch_count](int i) {
                    return i * batch_count + batch_i;
                };

                if (iteration < hyper_params.warm_up_iterations) {
                    optimizer.setLearningRate(hyper_params.learning_rate);
                } else {
                    dtype lr = hyper_params.learning_rate * sqrt(hyper_params.warm_up_iterations) /
                        sqrt(iteration);
                    optimizer.setLearningRate(lr);
                }
                if (batch_i % 10 == 5) {
                    cout << "lr:" << optimizer.getLearningRate() << endl;
                }

                vector<Node *> decoder_outputs;
                for (int i = 0; i < batch_size; ++i) {
                    shared_ptr<GraphBuilder> graph_builder(new GraphBuilder);
                    graph_builders.push_back(graph_builder);
                    int instance_index = getSentenceIndex(i);
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    conversation_pair_in_batch.push_back(train_conversation_pairs.at(
                                instance_index));
                    graph_builder->forward(graph, post_sentences.at(post_id), hyper_params,
                            model_params);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    Node *node = graph_builder->forwardDecoder(response_sentences.at(response_id),
                            hyper_params, model_params);
                    decoder_outputs.push_back(node);
                }
                profiler.EndCudaEvent();

                graph.forward();
                int word_sum = 0;
                for (int i = 0; i < batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int res_id = train_conversation_pairs.at(instance_index).response_id;
                    int size = response_sentences.at(res_id).size();
                    word_sum += size;
                }
                corpus_word_sum += word_sum;

                vector<vector<int>> total_word_ids;

                for (int i = 0; i < batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    vector<int> word_ids = toIds(response_sentences.at(response_id),
                            model_params.lookup_table);
                    total_word_ids.push_back(move(word_ids));
                }

                float loss = insnet::NLLLoss(decoder_outputs,
                        model_params.lookup_table.size(), total_word_ids, 1.0/ word_sum);
                loss_sum += loss * word_sum;

                auto predicted_ids = argmax(decoder_outputs, model_params.lookup_table.nVSize);
                for (int i = 0; i < predicted_ids.size(); ++i) {
                    analyze(predicted_ids.at(i), total_word_ids.at(i), *metric);
                }

                if (batch_i % 10 == 5) {
                    int instance_index = getSentenceIndex(0);
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    cout << "post:" << post_id << endl;
                    print(post_sentences.at(post_id));
                    cout << "golden answer:" << endl;
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    vector<int> word_ids = toIds(response_sentences.at(response_id),
                            model_params.lookup_table);
                    printWordIds(word_ids, model_params.lookup_table);
                    cout << fmt::format("loss:{}\n", loss);
                    cout << "output:" << endl;
                    printWordIds(predicted_ids.at(0), model_params.lookup_table);
                }
                if (batch_i % 10 == 5) {
                    cout << "ppl:" << exp(loss_sum / corpus_word_sum) << std::endl;
                    metric->print();
                }

                graph.backward();

//                if (default_config.check_grad) {
//                    auto loss_function = [&](const ConversationPair &conversation_pair) -> dtype {
//                        GraphBuilder graph_builder;
//                        Graph graph(false);

//                        graph_builder.forward(graph, post_sentences.at(conversation_pair.post_id),
//                                hyper_params, model_params, true);

//                        DecoderComponents decoder_components;
//                        graph_builder.forwardDecoder(graph, decoder_components,
//                                response_sentences.at(conversation_pair.response_id),
//                                hyper_params, model_params, true);

//                        graph.compute();

//                        vector<int> word_ids = toIds(response_sentences.at(
//                                    conversation_pair.response_id),
//                                model_params.decoder_lookup_table);
//                        vector<Node*> result_nodes = toNodePointers(
//                                decoder_components.wordvector_to_onehots);
//                        return maxLogProbabilityLoss(result_nodes, word_ids, 1.0 /
//                                word_ids.size()).first;
//                    };
//                    cout << format("checking grad - conversation_pair size:%1%") %
//                        conversation_pair_in_batch.size() << endl;
//                    grad_checker.check<ConversationPair>(loss_function, conversation_pair_in_batch,
//                            "");
//                }

                if (hyper_params.clip_grad > 1e4) {
                    optimizer.step();
                } else {
                    optimizer.step(hyper_params.clip_grad);
                }
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(stop - start);
                duration_count = 0.99 * duration_count + 0.01 * duration.count();
                if (batch_i % 10 == 5) {
                    cout << "duration:" << duration_count << endl;
                }

                ++iteration;
            }
            float perplex = metricTestPosts(hyper_params, model_params,
                    dev_post_and_responses, post_sentences, response_sentences);
            cout << "dev ppl:" << perplex << endl;


            cout << "loss_sum:" << loss_sum << endl;
            cout << "learning_rate now:" << hyper_params.learning_rate << endl;
            last_saved_model = saveModel(hyper_params, model_params,
                    default_config.output_model_file_prefix, epoch);

            profiler.EndCudaEvent();
            profiler.Print();
            profiler.SetEnabled(false);
        }
    } else {
        abort();
    }

    return 0;
}
