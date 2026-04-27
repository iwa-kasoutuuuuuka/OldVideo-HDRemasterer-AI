#include "gui.h"
#include "localization.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <nfd.hpp>
#include <filesystem>

int GUIManager::run() {
    NFD::Init();
    Localization& loc = Localization::getInstance();

    if (!glfwInit()) return -1;
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "OldVideo HDRemasterer AI Dashboard", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    // 日本語フォントの読み込み (Windows標準のMSゴシック)
    if (std::filesystem::exists("C:\\Windows\\Fonts\\msgothic.ttc")) {
        io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\msgothic.ttc", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    VideoProcessor::Config config;
    char input_buf[256] = "";
    char output_buf[256] = "";
    
    // スレッド同期用
    std::atomic<bool> processing{false};
    std::atomic<float> progress{0.0f};
    std::string status_msg = loc.s().status_idle;
    std::string current_operation = "";
    std::thread worker;

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
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), loc.s().title.c_str());
        
        ImGui::NextColumn();
        static int current_lang_idx = 0; // 0=Bilingual, 1=English
        const char* lang_items[] = { "JP / EN", "English" };
        ImGui::PushItemWidth(-1);
        if (ImGui::Combo("##Lang", &current_lang_idx, lang_items, IM_ARRAYSIZE(lang_items))) {
            loc.setLanguage(current_lang_idx == 0 ? Language::Bilingual : Language::English);
            status_msg = loc.s().status_idle; // Reset message on lang change for consistency
        }
        ImGui::PopItemWidth();
        ImGui::Columns(1);
        ImGui::Separator();

        ImGui::BeginDisabled(processing);
        
        ImGui::Text(loc.s().input_file.c_str());
        ImGui::InputText("##InputPath", input_buf, IM_ARRAYSIZE(input_buf));
        ImGui::SameLine();
        if (ImGui::Button((loc.s().browse + "##Input").c_str())) {
            nfdnchar_t* outPath = nullptr;
            nfdresult_t result = NFD::OpenDialog(outPath, nullptr, 0, nullptr);
            if (result == NFD_OKAY) {
                #ifdef _WIN32
                std::string conv(outPath, outPath + wcslen(outPath));
                #else
                std::string conv(outPath);
                #endif
                strncpy(input_buf, conv.c_str(), sizeof(input_buf)-1);
                NFD::FreePath(outPath);
            }
        }

        ImGui::Text(loc.s().output_file.c_str());
        ImGui::InputText("##OutputPath", output_buf, IM_ARRAYSIZE(output_buf));
        ImGui::SameLine();
        if (ImGui::Button((loc.s().browse + "##Output").c_str())) {
            nfdnchar_t* outPath = nullptr;
            nfdresult_t result = NFD::SaveDialog(outPath, nullptr, 0, nullptr, nullptr);
            if (result == NFD_OKAY) {
                #ifdef _WIN32
                std::string conv(outPath, outPath + wcslen(outPath));
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
            if (!std::filesystem::exists(in_path)) {
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
                progress = 0.0f;
                status_msg = "Processing...";

                // 別スレッドで実行
                if (worker.joinable()) worker.join();
                worker = std::thread([&, config]() mutable {
                    VideoProcessor processor;
                    config.progress_callback = [&](float p, std::string msg) {
                        progress = p;
                        current_operation = msg;
                    };
                    
                    bool success = processor.run(config);
                    
                    processing = false;
                    status_msg = success ? loc.s().status_success : loc.s().status_error;
                    current_operation = "";
                });
            } // end validation
        }

        // プログレスバー
        char buf[64];
        sprintf(buf, "%d%%", (int)(progress * 100));
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), buf);

        // ステータス表示
        ImGui::Spacing();
        if (processing) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s %s", loc.s().running_prefix.c_str(), current_operation.c_str());
        } else {
            ImGui::Text("%s: %s", loc.s().status.c_str(), status_msg.c_str());
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

    if (worker.joinable()) worker.join();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    NFD::Quit();

    return 0;
}
