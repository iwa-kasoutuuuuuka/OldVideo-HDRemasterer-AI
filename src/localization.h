#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include <string>

/**
 * @brief 言語管理クラス / Language Management Class
 */
enum class Language {
    Bilingual, // 日本語 / English
    English    // English Only
};

struct UIStrings {
    std::string title;
    std::string input_file;
    std::string output_file;
    std::string browse;
    std::string scale;
    std::string tile_size;
    std::string ai_model;
    std::string model_high;
    std::string model_fast;
    std::string enable_stab;
    std::string enable_interp;
    std::string start_btn;
    std::string processing_btn;
    std::string stop_btn;
    std::string status;
    std::string status_idle;
    std::string status_success;
    std::string status_error;
    std::string err_input_missing;
    std::string err_output_empty;
    std::string running_prefix;
    std::string lang_switch;
};

class Localization {
public:
    static Localization& getInstance() {
        static Localization instance;
        return instance;
    }

    void setLanguage(Language lang) {
        currentLang = lang;
        updateStrings();
    }

    Language getLanguage() const { return currentLang; }
    const UIStrings& s() const { return strings; }

private:
    Localization() { setLanguage(Language::Bilingual); }
    Language currentLang = Language::Bilingual;
    UIStrings strings;

    void updateStrings() {
        if (currentLang == Language::Bilingual) {
            strings.title = "OldVideo HDRemasterer AI - リマスター・ダッシュボード / Dashboard";
            strings.input_file = "入力ファイル / Input File";
            strings.output_file = "出力ファイル / Output File";
            strings.browse = "参照 / Browse";
            strings.scale = "拡大率 / Scale";
            strings.tile_size = "タイルサイズ / Tile Size";
            strings.ai_model = "AIモデル / AI Model";
            strings.model_high = "高品質 / High Quality (realesrgan-x4plus)";
            strings.model_fast = "高速 / Fast (animevideov3)";
            strings.enable_stab = "手ブレ補正を有効化 / Enable Stabilization";
            strings.enable_interp = "フレーム補間 (RIFE) を有効化 / Enable Interpolation (60fps)";
            strings.start_btn = "リマスター開始 / Start Remaster";
            strings.processing_btn = "処理中... / Processing...";
            strings.stop_btn = "停止 / Stop";
            strings.status = "ステータス / Status";
            strings.status_idle = "動画ファイルを選択してください / Please select a video file";
            strings.status_success = "成功: 完了しました！ / Success: Completed!";
            strings.status_error = "エラー: 失敗しました / Error: Failed";
            strings.err_input_missing = "エラー: 入力ファイルが存在しません / Error: Input file not found";
            strings.err_output_empty = "エラー: 出力パスが空です / Error: Output path is empty";
            strings.running_prefix = "[実行中] / [Running]";
            strings.lang_switch = "言語 / Language";
        } else {
            strings.title = "OldVideo HDRemasterer AI - Dashboard";
            strings.input_file = "Input File";
            strings.output_file = "Output File";
            strings.browse = "Browse";
            strings.scale = "Scale";
            strings.tile_size = "Tile Size";
            strings.ai_model = "AI Model";
            strings.model_high = "High Quality (realesrgan-x4plus)";
            strings.model_fast = "Fast (animevideov3)";
            strings.enable_stab = "Enable Stabilization";
            strings.enable_interp = "Enable Interpolation (60fps)";
            strings.start_btn = "Start Remaster";
            strings.processing_btn = "Processing...";
            strings.stop_btn = "Stop";
            strings.status = "Status";
            strings.status_idle = "Please select a video file";
            strings.status_success = "Success: Completed!";
            strings.status_error = "Error: Failed";
            strings.err_input_missing = "Error: Input file not found";
            strings.err_output_empty = "Error: Output path is empty";
            strings.running_prefix = "[Running]";
            strings.lang_switch = "Language";
        }
    }
};

#endif // LOCALIZATION_H
