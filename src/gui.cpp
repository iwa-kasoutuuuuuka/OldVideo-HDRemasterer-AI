#include "gui.h"
#include "localization.h"
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
// Windows SEH (構造化例外処理) を使ってアクセス違反等をキャッチ
static LONG WINAPI crashHandler(EXCEPTION_POINTERS* ep) {
    std::cerr << "[FATAL] Windows構造化例外が発生しました。コード: 0x"
              << std::hex << ep->ExceptionRecord->ExceptionCode
              << std::dec << std::endl << std::flush;
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

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
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "OldVideo HDRemasterer AI Dashboard", NULL, NULL);
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

    std::cout << "[GUI] GPU 検出数: " << ncnn::get_gpu_count() << std::endl << std::flush;
    std::cout << "[GUI] メインループ開始" << std::endl << std::flush;

    try {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

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

        ImGui::BeginDisabled(processing);
        
        ImGui::Text("%s", loc.s().input_file.c_str());
        ImGui::InputText("##InputPath", input_buf, IM_ARRAYSIZE(input_buf));
        ImGui::SameLine();
        if (ImGui::Button((loc.s().browse + "##Input").c_str())) {
            nfdnchar_t* outPath = nullptr;
            nfdresult_t result = NFD::OpenDialog(outPath, nullptr, 0, nullptr);
            if (result == NFD_OKAY) {
                #ifdef _WIN32
                // wchar_t (UTF-16) to UTF-8
                int size_needed = WideCharToMultiByte(CP_UTF8, 0, outPath, -1, NULL, 0, NULL, NULL);
                std::string conv(size_needed, 0);
                WideCharToMultiByte(CP_UTF8, 0, outPath, -1, &conv[0], size_needed, NULL, NULL);
                conv.erase(conv.find('\0')); // Remove null terminator
                #else
                std::string conv(outPath);
                #endif
                strncpy(input_buf, conv.c_str(), sizeof(input_buf)-1);
                NFD::FreePath(outPath);
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
                NFD::FreePath(outPath);
            }
        }

        ImGui::SliderInt(loc.s().scale.c_str(), &config.scale, 2, 4);
        if (config.scale == 3) config.scale = 4; 

        ImGui::SliderInt(loc.s().tile_size.c_str(), &config.tile_size, 0, 1024);
        
        // モデル選択
        static int model_mode = 0; // 0=高品質, 1=高速
        const char* model_items[] = { loc.s().model_high.c_str(), loc.s().model_fast.c_str() };
        ImGui::Combo(loc.s().ai_model.c_str(), &model_mode, model_items, IM_ARRAYSIZE(model_items));
        
        static bool enable_stab = false;
        ImGui::Checkbox(loc.s().enable_stab.c_str(), &enable_stab);
        
        static bool enable_interp = false;
        ImGui::Checkbox(loc.s().enable_interp.c_str(), &enable_interp);
        ImGui::EndDisabled();

        ImGui::Separator();

        std::string btn_label = processing ? loc.s().processing_btn : loc.s().start_btn;
        if (ImGui::Button(btn_label.c_str(), ImVec2(-1.0f, 40.0f)) && !processing) {
            std::string in_path(input_buf);
            bool exists = false;
            #ifdef _WIN32
            int wsize = MultiByteToWideChar(CP_UTF8, 0, in_path.c_str(), -1, NULL, 0);
            std::wstring win_path(wsize, 0);
            MultiByteToWideChar(CP_UTF8, 0, in_path.c_str(), -1, &win_path[0], wsize);
            exists = std::filesystem::exists(win_path);
            #else
            exists = std::filesystem::exists(in_path);
            #endif

            if (!exists) {
                status_msg = loc.s().err_input_missing;
            } else if (strlen(output_buf) == 0) {
                status_msg = loc.s().err_output_empty;
            } else {
                config.input_path = input_buf;
                config.output_path = output_buf;
                config.enable_stabilization = enable_stab;
                config.enable_interpolation = enable_interp;
                config.use_fast_model = (model_mode == 1);
                
                processing = true;
                should_stop = false;
                config.stop_flag = &should_stop;
                progress = 0.0f;
                status_msg = "Processing...";

                // 別スレッドで実行
                if (worker.joinable()) worker.join();
                worker = std::thread([&, config]() mutable {
                    try {
                        std::cout << "[WORKER] スレッド開始 / Worker thread started" << std::endl << std::flush;
                        
                        auto processor = std::make_unique<VideoProcessor>();
                        std::cout << "[WORKER] VideoProcessor 初期化完了" << std::endl << std::flush;
                        
                        config.progress_callback = [&](float p, std::string msg) {
                            progress = p;
                            std::lock_guard<std::mutex> lk(gui_mtx);
                            current_operation = msg;
                        };
                        
                        std::cout << "[WORKER] 処理開始: " << config.input_path << std::endl << std::flush;
                        bool success = processor->run(config);
                        std::cout << "[WORKER] 処理終了: " << (success ? "成功" : "失敗") << std::endl << std::flush;
                        
                        {
                            std::lock_guard<std::mutex> lk(gui_mtx);
                            if (should_stop) {
                                status_msg = "Stopped";
                            } else {
                                status_msg = success ? loc.s().status_success : loc.s().status_error;
                            }
                            current_operation = "";
                        }
                        processing = false;
                    } catch (const std::exception& e) {
                        std::cerr << "[CRASH] C++ 例外: " << e.what() << std::endl << std::flush;
                        {
                            std::lock_guard<std::mutex> lk(gui_mtx);
                            status_msg = std::string("Error: ") + e.what();
                            current_operation = "";
                        }
                        processing = false;
                    } catch (...) {
                        std::cerr << "[CRASH] 不明な例外が発生しました" << std::endl << std::flush;
                        {
                            std::lock_guard<std::mutex> lk(gui_mtx);
                            status_msg = "Error: Unknown crash";
                            current_operation = "";
                        }
                        processing = false;
                    }
                });
            } // end validation
        }

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
        sprintf(buf, "%d%%", (int)(progress * 100));
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

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    NFD::Quit();

    return 0;
}
