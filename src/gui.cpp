#include "gui.h"
#include "localization.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <nfd.hpp>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
static LONG WINAPI crashHandler(EXCEPTION_POINTERS* ep) {
    std::cerr << "[FATAL] Windows構造化例外が発生しました。コード: 0x"
              << std::hex << ep->ExceptionRecord->ExceptionCode
              << std::dec << std::endl << std::flush;
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

// OpenCV Mat を OpenGL テクスチャに変換するヘルパー関数
GLuint MatToTexture(const cv::Mat& mat) {
    if (mat.empty()) return 0;
    
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureID;
}

struct BatchItem {
    std::string input_path;
    std::string output_path;
    int scale = 4;
    int tile_size = 0;
    std::string model_name;
    bool enable_stab = false;
    bool enable_interp = false;
    bool enable_trim = false;
    double trim_start = 0.0;
    double trim_duration = 10.0;
};

int GUIManager::run() {
    std::cout << "[GUI] run() 開始" << std::endl << std::flush;

#ifdef _WIN32
    SetUnhandledExceptionFilter(crashHandler);
#endif

    NFD::Init();
    std::cout << "[GUI] NFD 初期化完了" << std::endl << std::flush;

    Localization& loc = Localization::getInstance();

    if (!glfwInit()) {
        std::cerr << "[GUI] glfwInit 失敗" << std::endl << std::flush;
        return -1;
    }
    std::cout << "[GUI] GLFW 初期化完了" << std::endl << std::flush;
    
    GLFWwindow* window = glfwCreateWindow(900, 700, "OldVideo HDRemasterer AI Dashboard", NULL, NULL);
    if (!window) {
        std::cerr << "[GUI] ウィンドウ作成失敗" << std::endl << std::flush;
        glfwTerminate();
        return -1;
    }
    std::cout << "[GUI] ウィンドウ作成完了" << std::endl << std::flush;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    std::cout << "[GUI] ImGui コンテキスト作成完了" << std::endl << std::flush;

    // 日本語フォントの読み込み (Windows標準のMSゴシック)
    if (std::filesystem::exists("C:\\Windows\\Fonts\\msgothic.ttc")) {
        io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\msgothic.ttc", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
        std::cout << "[GUI] 日本語フォントロード完了" << std::endl << std::flush;
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    std::cout << "[GUI] ImGui バックエンド初期化完了" << std::endl << std::flush;

    // FFmpegチェック
    std::atomic<bool> ffmpeg_available{ utils::check_ffmpeg_available() };
    std::atomic<bool> downloading_ffmpeg{ false };

    VideoProcessor::Config config;
    char input_buf[256] = "";
    char output_buf[256] = "";
    
    // スレッド同期用
    std::mutex gui_mtx;
    std::atomic<bool> processing{false};
    bool should_stop{false};
    std::atomic<float> progress{0.0f};
    std::string status_msg = loc.s().status_idle;
    std::string current_operation = "";
    std::thread worker;
    std::thread preview_worker; // 比較プレビュー用スレッドを保持
    std::thread ffmpeg_worker;  // FFmpegダウンロード用スレッドを保持

    // モデルリスト
    const std::vector<std::string> model_names = {
        "realesrgan-x4plus",
        "realesrgan-x4plus-anime",
        "realesr-animevideov3-x4",
        "realesr-animevideov3-x3",
        "realesr-animevideov3-x2"
    };
    int selected_model_idx = 0;

    // トリミングパラメータ
    bool enable_trim = false;
    float trim_start = 0.0f;
    float trim_duration = 10.0f;

    // バッチキュー
    std::vector<BatchItem> batch_queue;
    char batch_input_buf[256] = "";
    char batch_output_buf[256] = "";

    // 比較プレビュー用
    int preview_frame_idx = 0;
    int preview_total_frames = 100;
    GLuint before_tex = 0;
    GLuint after_tex = 0;
    float split_ratio = 0.5f;
    std::mutex preview_mtx;
    cv::Mat before_mat_pending;
    cv::Mat after_mat_pending;
    std::atomic<bool> preview_updated{ false };
    std::atomic<bool> generating_preview{ false };

    std::cout << "[GUI] GPU 検出数: " << ncnn::get_gpu_count() << std::endl << std::flush;
    std::cout << "[GUI] メインループ開始" << std::endl << std::flush;

    try {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // 非同期で生成されたプレビューテクスチャのメインスレッド転送
            if (preview_updated) {
                std::lock_guard<std::mutex> lk(preview_mtx);
                if (before_tex) glDeleteTextures(1, &before_tex);
                if (after_tex) glDeleteTextures(1, &after_tex);
                
                before_tex = MatToTexture(before_mat_pending);
                after_tex = MatToTexture(after_mat_pending);
                preview_updated = false;
                generating_preview = false;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::Begin("Dashboard", NULL, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

            // --- Header & Language Switcher ---
            ImGui::Columns(2, "Header", false);
            ImGui::SetColumnWidth(0, io.DisplaySize.x - 150.0f);
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "%s", loc.s().title.c_str());
            
            ImGui::NextColumn();
            static int current_lang_idx = 0; // 0=Bilingual, 1=English
            const char* lang_items[] = { "JP / EN", "English" };
            ImGui::PushItemWidth(-1);
            if (ImGui::Combo("##Lang", &current_lang_idx, lang_items, IM_ARRAYSIZE(lang_items))) {
                loc.setLanguage(current_lang_idx == 0 ? Language::Bilingual : Language::English);
                status_msg = loc.s().status_idle;
            }
            ImGui::PopItemWidth();
            ImGui::Columns(1);
            ImGui::Separator();

            // --- FFmpeg Warning Banner ---
            if (!ffmpeg_available) {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.3f, 0.1f, 0.1f, 1.0f));
                ImGui::BeginChild("FFmpegWarning", ImVec2(-1.0f, 65.0f), true);
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), 
                    "FFmpeg が見つかりません。音声マージ等の処理には FFmpeg が必要です。 / FFmpeg was not found.");
                
                if (downloading_ffmpeg) {
                    ImGui::Text("FFmpeg をダウンロード中... / Downloading FFmpeg...");
                } else {
                    ImGui::SameLine();
                    if (ImGui::Button("FFmpeg を自動ダウンロード / Download FFmpeg")) {
                        downloading_ffmpeg = true;
                        if (ffmpeg_worker.joinable()) {
                            ffmpeg_worker.join();
                        }
                        ffmpeg_worker = std::thread([&]() {
                            std::cout << "[FFMPEG] ダウンロード開始" << std::endl;
                            std::system("powershell -ExecutionPolicy Bypass -File scripts/download_ffmpeg.ps1");
                            ffmpeg_available = utils::check_ffmpeg_available();
                            downloading_ffmpeg = false;
                            std::cout << "[FFMPEG] ダウンロードおよび存在確認完了: " << (ffmpeg_available ? "成功" : "失敗") << std::endl;
                        });
                    }
                }
                ImGui::EndChild();
                ImGui::PopStyleColor();
                ImGui::Spacing();
            }

            // --- Tab Bar ---
            if (ImGui::BeginTabBar("MainTabBar")) {
                
                // ==================== タブ1: 単一ファイル処理 ====================
                if (ImGui::BeginTabItem("単一リマスター / Single Remaster")) {
                    ImGui::BeginDisabled(processing);
                    
                    ImGui::Text("%s", loc.s().input_file.c_str());
                    ImGui::InputText("##InputPath", input_buf, IM_ARRAYSIZE(input_buf));
                    ImGui::SameLine();
                    if (ImGui::Button((loc.s().browse + "##Input").c_str())) {
                        nfdnchar_t* outPath = nullptr;
                        nfdresult_t result = NFD::OpenDialog(outPath, nullptr, 0, nullptr);
                        if (result == NFD_OKAY) {
                            #ifdef _WIN32
                            int size_needed = WideCharToMultiByte(CP_UTF8, 0, outPath, -1, NULL, 0, NULL, NULL);
                            std::string conv(size_needed, 0);
                            WideCharToMultiByte(CP_UTF8, 0, outPath, -1, &conv[0], size_needed, NULL, NULL);
                            conv.erase(conv.find('\0'));
                            #else
                            std::string conv(outPath);
                            #endif
                            strncpy(input_buf, conv.c_str(), sizeof(input_buf)-1);
                            input_buf[sizeof(input_buf)-1] = '\0';
                            NFD::FreePath(outPath);

                            // 総フレーム数の更新を試みる
                            cv::VideoCapture temp_cap(input_buf);
                            if (temp_cap.isOpened()) {
                                preview_total_frames = static_cast<int>(temp_cap.get(cv::CAP_PROP_FRAME_COUNT));
                                temp_cap.release();
                            }
                        }
                    }

                    ImGui::Text("%s", loc.s().output_file.c_str());
                    ImGui::InputText("##OutputPath", output_buf, IM_ARRAYSIZE(output_buf));
                    ImGui::SameLine();
                    if (ImGui::Button((loc.s().browse + "##Output").c_str())) {
                        nfdnchar_t* outPath = nullptr;
                        nfdresult_t result = NFD::SaveDialog(outPath, nullptr, 0, nullptr, nullptr);
                        if (result == NFD_OKAY) {
                            #ifdef _WIN32
                            int size_needed = WideCharToMultiByte(CP_UTF8, 0, outPath, -1, NULL, 0, NULL, NULL);
                            std::string conv(size_needed, 0);
                            WideCharToMultiByte(CP_UTF8, 0, outPath, -1, &conv[0], size_needed, NULL, NULL);
                            conv.erase(conv.find('\0'));
                            #else
                            std::string conv(outPath);
                            #endif
                            strncpy(output_buf, conv.c_str(), sizeof(output_buf)-1);
                            output_buf[sizeof(output_buf)-1] = '\0';
                            NFD::FreePath(outPath);
                        }
                    }

                    ImGui::SliderInt(loc.s().scale.c_str(), &config.scale, 2, 4);
                    if (config.scale == 3) config.scale = 4; 

                    ImGui::SliderInt(loc.s().tile_size.c_str(), &config.tile_size, 0, 1024);
                    
                    // AIモデルの選択
                    std::vector<const char*> model_items;
                    for (const auto& name : model_names) {
                        model_items.push_back(name.c_str());
                    }
                    ImGui::Combo("AIモデル / Model", &selected_model_idx, model_items.data(), static_cast<int>(model_items.size()));

                    static bool enable_stab = false;
                    ImGui::Checkbox(loc.s().enable_stab.c_str(), &enable_stab);
                    
                    static bool enable_interp = false;
                    ImGui::Checkbox(loc.s().enable_interp.c_str(), &enable_interp);

                    // トリミング設定
                    ImGui::Checkbox("部分処理（トリミング）を有効化 / Enable Trim", &enable_trim);
                    if (enable_trim) {
                        ImGui::SliderFloat("開始位置 (秒) / Start (s)", &trim_start, 0.0f, 3600.0f);
                        ImGui::SliderFloat("処理時間 (秒) / Duration (s)", &trim_duration, 1.0f, 120.0f);
                    }

                    ImGui::EndDisabled();

                    ImGui::Separator();

                    std::string btn_label = processing ? loc.s().processing_btn : loc.s().start_btn;
                    if (ImGui::Button(btn_label.c_str(), ImVec2(-1.0f, 40.0f)) && !processing) {
                        std::string in_path(input_buf);
                        bool exists = std::filesystem::exists(in_path);

                        if (!exists) {
                            status_msg = loc.s().err_input_missing;
                        } else if (strlen(output_buf) == 0) {
                            status_msg = loc.s().err_output_empty;
                        } else {
                            config.input_path = input_buf;
                            config.output_path = output_buf;
                            config.enable_stabilization = enable_stab;
                            config.enable_interpolation = enable_interp;
                            config.model_name = model_names[selected_model_idx];
                            config.enable_trim = enable_trim;
                            config.trim_start_sec = trim_start;
                            config.trim_duration_sec = trim_duration;
                            
                            processing = true;
                            should_stop = false;
                            config.stop_flag = &should_stop;
                            progress = 0.0f;
                            status_msg = "Processing...";

                            if (worker.joinable()) worker.join();
                            worker = std::thread([&, config]() mutable {
                                try {
                                    auto processor = std::make_unique<VideoProcessor>();
                                    config.progress_callback = [&](float p, std::string msg) {
                                        progress = p;
                                        std::lock_guard<std::mutex> lk(gui_mtx);
                                        current_operation = msg;
                                    };
                                    bool success = processor->run(config);
                                    {
                                        std::lock_guard<std::mutex> lk(gui_mtx);
                                        if (should_stop) status_msg = "Stopped";
                                        else status_msg = success ? loc.s().status_success : loc.s().status_error;
                                        current_operation = "";
                                    }
                                    processing = false;
                                } catch (const std::exception& e) {
                                    std::lock_guard<std::mutex> lk(gui_mtx);
                                    status_msg = std::string("Error: ") + e.what();
                                    processing = false;
                                }
                            });
                        }
                    }
                    ImGui::EndTabItem();
                }

                // ==================== タブ2: 一括処理 (バッチ) ====================
                if (ImGui::BeginTabItem("一括処理 / Batch Process")) {
                    ImGui::BeginDisabled(processing);
                    
                    ImGui::Text("一括入力ファイル / Input File");
                    ImGui::InputText("##BatchInputPath", batch_input_buf, IM_ARRAYSIZE(batch_input_buf));
                    ImGui::SameLine();
                    if (ImGui::Button("参照 / Browse##BatchInput")) {
                        nfdnchar_t* outPath = nullptr;
                        nfdresult_t result = NFD::OpenDialog(outPath, nullptr, 0, nullptr);
                        if (result == NFD_OKAY) {
                            #ifdef _WIN32
                            int size_needed = WideCharToMultiByte(CP_UTF8, 0, outPath, -1, NULL, 0, NULL, NULL);
                            std::string conv(size_needed, 0);
                            WideCharToMultiByte(CP_UTF8, 0, outPath, -1, &conv[0], size_needed, NULL, NULL);
                            conv.erase(conv.find('\0'));
                            #else
                            std::string conv(outPath);
                            #endif
                            strncpy(batch_input_buf, conv.c_str(), sizeof(batch_input_buf)-1);
                            batch_input_buf[sizeof(batch_input_buf)-1] = '\0';
                            NFD::FreePath(outPath);
                        }
                    }

                    ImGui::Text("一括出力ファイル / Output File");
                    ImGui::InputText("##BatchOutputPath", batch_output_buf, IM_ARRAYSIZE(batch_output_buf));
                    ImGui::SameLine();
                    if (ImGui::Button("参照 / Browse##BatchOutput")) {
                        nfdnchar_t* outPath = nullptr;
                        nfdresult_t result = NFD::SaveDialog(outPath, nullptr, 0, nullptr, nullptr);
                        if (result == NFD_OKAY) {
                            #ifdef _WIN32
                            int size_needed = WideCharToMultiByte(CP_UTF8, 0, outPath, -1, NULL, 0, NULL, NULL);
                            std::string conv(size_needed, 0);
                            WideCharToMultiByte(CP_UTF8, 0, outPath, -1, &conv[0], size_needed, NULL, NULL);
                            conv.erase(conv.find('\0'));
                            #else
                            std::string conv(outPath);
                            #endif
                            strncpy(batch_output_buf, conv.c_str(), sizeof(batch_output_buf)-1);
                            batch_output_buf[sizeof(batch_output_buf)-1] = '\0';
                            NFD::FreePath(outPath);
                        }
                    }

                    static bool batch_enable_stab = false;
                    static bool batch_enable_interp = false;
                    ImGui::Checkbox("手ブレ補正 / Stabilization##Batch", &batch_enable_stab);
                    ImGui::SameLine();
                    ImGui::Checkbox("フレーム補間 / Interpolation##Batch", &batch_enable_interp);

                    if (ImGui::Button("キューに追加 / Add to Queue")) {
                        if (std::filesystem::exists(batch_input_buf) && strlen(batch_output_buf) > 0) {
                            BatchItem item;
                            item.input_path = batch_input_buf;
                            item.output_path = batch_output_buf;
                            item.scale = config.scale;
                            item.tile_size = config.tile_size;
                            item.model_name = model_names[selected_model_idx];
                            item.enable_stab = batch_enable_stab;
                            item.enable_interp = batch_enable_interp;
                            item.enable_trim = enable_trim;
                            item.trim_start = trim_start;
                            item.trim_duration = trim_duration;
                            batch_queue.push_back(item);

                            // リセット
                            batch_input_buf[0] = '\0';
                            batch_output_buf[0] = '\0';
                        }
                    }

                    ImGui::EndDisabled();

                    ImGui::Spacing();
                    ImGui::Text("待機キュー一覧 / Queue List:");
                    ImGui::BeginChild("QueueListChild", ImVec2(-1.0f, 150.0f), true);
                    for (size_t i = 0; i < batch_queue.size(); ++i) {
                        ImGui::Text("[%d] %s -> %s (%s)", (int)i + 1, 
                            std::filesystem::path(batch_queue[i].input_path).filename().string().c_str(),
                            std::filesystem::path(batch_queue[i].output_path).filename().string().c_str(),
                            batch_queue[i].model_name.c_str());
                        ImGui::SameLine(ImGui::GetWindowWidth() - 70.0f);
                        ImGui::BeginDisabled(processing);
                        if (ImGui::Button(("削除##" + std::to_string(i)).c_str())) {
                            batch_queue.erase(batch_queue.begin() + i);
                        }
                        ImGui::EndDisabled();
                    }
                    ImGui::EndChild();

                    ImGui::Separator();

                    if (ImGui::Button(processing ? "処理中... / Processing..." : "一括処理を開始 / Start Batch", ImVec2(-1.0f, 40.0f)) && !processing && !batch_queue.empty()) {
                        processing = true;
                        should_stop = false;
                        progress = 0.0f;
                        status_msg = "Processing batch...";

                        if (worker.joinable()) worker.join();
                        worker = std::thread([&, batch_queue]() {
                            int total_tasks = static_cast<int>(batch_queue.size());
                            for (int i = 0; i < total_tasks; ++i) {
                                if (should_stop) break;

                                const auto& item = batch_queue[i];
                                VideoProcessor::Config item_config;
                                item_config.input_path = item.input_path;
                                item_config.output_path = item.output_path;
                                item_config.scale = item.scale;
                                item_config.tile_size = item.tile_size;
                                item_config.model_name = item.model_name;
                                item_config.enable_stabilization = item.enable_stab;
                                item_config.enable_interpolation = item.enable_interp;
                                item_config.enable_trim = item.enable_trim;
                                item_config.trim_start_sec = item.trim_start;
                                item_config.trim_duration_sec = item.trim_duration;
                                item_config.stop_flag = &should_stop;

                                item_config.progress_callback = [&](float p, std::string msg) {
                                    progress = (static_cast<float>(i) + p) / static_cast<float>(total_tasks);
                                    std::lock_guard<std::mutex> lk(gui_mtx);
                                    current_operation = "[Task " + std::to_string(i + 1) + "/" + std::to_string(total_tasks) + "] " + msg;
                                };

                                auto processor = std::make_unique<VideoProcessor>();
                                bool success = processor->run(item_config);
                                if (!success) {
                                    std::lock_guard<std::mutex> lk(gui_mtx);
                                    status_msg = "Task " + std::to_string(i + 1) + " failed.";
                                }
                            }
                            {
                                std::lock_guard<std::mutex> lk(gui_mtx);
                                if (should_stop) status_msg = "Stopped";
                                else status_msg = "Batch Process Completed";
                                current_operation = "";
                            }
                            processing = false;
                        });
                    }
                    ImGui::EndTabItem();
                }

                // ==================== タブ3: 比較プレビュー ====================
                if (ImGui::BeginTabItem("比較プレビュー / Preview Comparison")) {
                    ImGui::Text("プレビュー表示する動画 / Preview Target");
                    if (ImGui::InputText("##PreviewInputPath", input_buf, IM_ARRAYSIZE(input_buf))) {
                        cv::VideoCapture temp_cap(input_buf);
                        if (temp_cap.isOpened()) {
                            preview_total_frames = static_cast<int>(temp_cap.get(cv::CAP_PROP_FRAME_COUNT));
                            temp_cap.release();
                        }
                    }

                    ImGui::SliderInt("表示フレーム指定 / Target Frame", &preview_frame_idx, 0, preview_total_frames - 1);
                    ImGui::SliderFloat("比較位置スライダー / Split Slider", &split_ratio, 0.0f, 1.0f);

                    ImGui::BeginDisabled(generating_preview);
                    if (ImGui::Button("プレビューを生成 / Generate Preview") && strlen(input_buf) > 0) {
                        generating_preview = true;
                        if (preview_worker.joinable()) preview_worker.join();
                        
                        preview_worker = std::thread([&, input_path = std::string(input_buf), frame_idx = preview_frame_idx, model_name = model_names[selected_model_idx], scale = config.scale, tile_size = config.tile_size]() {
                            cv::VideoCapture cap(input_path);
                            if (cap.isOpened()) {
                                cap.set(cv::CAP_PROP_POS_FRAMES, frame_idx);
                                cv::Mat raw_frame;
                                cap.read(raw_frame);
                                cap.release();

                                if (!raw_frame.empty()) {
                                    {
                                        std::lock_guard<std::mutex> lk(preview_mtx);
                                        before_mat_pending = raw_frame.clone();
                                    }

                                    // 超解像の適用 (プレビュー生成)
                                    Upscaler temp_upscaler;
                                    if (temp_upscaler.load("models", scale, tile_size, model_name)) {
                                        cv::Mat upscaled_frame;
                                        if (temp_upscaler.process(raw_frame, upscaled_frame)) {
                                            std::lock_guard<std::mutex> lk(preview_mtx);
                                            after_mat_pending = upscaled_frame.clone();
                                            preview_updated = true;
                                        }
                                    }
                                }
                            } else {
                                std::cerr << "[ERROR] プレビュー動画ファイルを開けませんでした。 / Failed to open preview video." << std::endl;
                            }
                            if (!preview_updated) generating_preview = false;
                        });
                    }
                    ImGui::EndDisabled();

                    if (generating_preview) {
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "プレビュー処理をバックグラウンドで処理中...");
                    }

                    ImGui::Spacing();
                    // 比較描画エリア
                    ImVec2 screen_pos = ImGui::GetCursorScreenPos();
                    ImVec2 size(640, 360);
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();

                    // 黒の背景
                    draw_list->AddRectFilled(screen_pos, ImVec2(screen_pos.x + size.x, screen_pos.y + size.y), IM_COL32(20, 20, 20, 255));

                    if (before_tex && after_tex) {
                        float split_x = screen_pos.x + size.x * split_ratio;

                        // 左半分: Before (リマスター前)
                        draw_list->PushClipRect(screen_pos, ImVec2(split_x, screen_pos.y + size.y), true);
                        draw_list->AddImage((void*)(intptr_t)before_tex, screen_pos, ImVec2(screen_pos.x + size.x, screen_pos.y + size.y));
                        draw_list->PopClipRect();

                        // 右半分: After (リマスター後)
                        draw_list->PushClipRect(ImVec2(split_x, screen_pos.y), ImVec2(screen_pos.x + size.x, screen_pos.y + size.y), true);
                        draw_list->AddImage((void*)(intptr_t)after_tex, screen_pos, ImVec2(screen_pos.x + size.x, screen_pos.y + size.y));
                        draw_list->PopClipRect();

                        // 境界分割線
                        draw_list->AddLine(ImVec2(split_x, screen_pos.y), ImVec2(split_x, screen_pos.y + size.y), IM_COL32(0, 255, 255, 255), 2.0f);
                    } else {
                        draw_list->AddText(ImVec2(screen_pos.x + 200, screen_pos.y + 170), IM_COL32(150, 150, 150, 255), 
                            "プレビュー画像がありません。上のボタンを押してください。 / No preview loaded.");
                    }
                    ImGui::Dummy(size);

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            ImGui::Separator();

            if (processing) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
                if (ImGui::Button(loc.s().stop_btn.c_str(), ImVec2(-1.0f, 40.0f))) {
                    should_stop = true;
                }
                ImGui::PopStyleColor(2);
            }

            // プログレスバー
            char buf[64];
            snprintf(buf, sizeof(buf), "%d%%", (int)(progress * 100));
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), buf);

            // ステータス表示
            ImGui::Spacing();
            {
                std::lock_guard<std::mutex> lk(gui_mtx);
                if (processing) {
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s %s", loc.s().running_prefix.c_str(), current_operation.c_str());
                } else {
                    ImGui::Text("%s: %s", loc.s().status.c_str(), status_msg.c_str());
                }
            }

            ImGui::End();

            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }
    } catch (const std::exception& e) {
        std::cerr << "[CRASH] メインループ内 C++ 例外: " << e.what() << std::endl << std::flush;
    } catch (...) {
        std::cerr << "[CRASH] メインループ内 不明な例外" << std::endl << std::flush;
    }

    std::cout << "[GUI] メインループ終了" << std::endl << std::flush;

    if (worker.joinable()) worker.join();
    if (preview_worker.joinable()) preview_worker.join();
    if (ffmpeg_worker.joinable()) ffmpeg_worker.join();

    if (before_tex) glDeleteTextures(1, &before_tex);
    if (after_tex) glDeleteTextures(1, &after_tex);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    NFD::Quit();

    return 0;
}
