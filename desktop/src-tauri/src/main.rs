use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use tauri::Manager;

struct Session {
    child: std::process::Child,
    reader: BufReader<std::process::ChildStdout>,
}

fn config_file(app: &tauri::AppHandle) -> PathBuf {
    if let Ok(dir) = std::env::var("PHOENX_GUI_CONFIG") {
        return PathBuf::from(dir).join("python-path");
    }
    app.path()
        .app_config_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("python-path")
}

fn can_import(python: &Path) -> bool {
    Command::new(python)
        .args(["-c", "import phoenx"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn chosen_python(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("PHOENX_PYTHON") {
        let file = PathBuf::from(&path);
        if file.is_file() {
            return Ok(file);
        }
    }
    let saved = std::fs::read_to_string(config_file(app)).map_err(|_| {
        "Choose the python.exe from the environment where PhoenX is installed.".to_string()
    })?;
    Ok(PathBuf::from(saved.trim()))
}

#[tauri::command]
fn python_status(app: tauri::AppHandle) -> serde_json::Value {
    if let Ok(path) = std::env::var("PHOENX_PYTHON") {
        if Path::new(&path).is_file() {
            return serde_json::json!({ "ready": true, "path": path });
        }
    }
    if let Ok(text) = std::fs::read_to_string(config_file(&app)) {
        let path = PathBuf::from(text.trim());
        if path.is_file() && can_import(&path) {
            return serde_json::json!({ "ready": true, "path": path });
        }
    }
    serde_json::json!({ "ready": false })
}

#[tauri::command]
fn set_python(app: tauri::AppHandle, path: String) -> Result<(), String> {
    let python = PathBuf::from(&path);
    if !python.is_file() || !can_import(&python) {
        return Err(
            "That Python cannot import phoenx. Choose the interpreter from the PhoenX environment."
                .into(),
        );
    }
    let file = config_file(&app);
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    std::fs::write(file, path).map_err(|err| err.to_string())
}

// async: runs on a worker thread, so a slow session reply never freezes
// the window (sync commands run on the main thread).
#[tauri::command(async)]
fn session_call(
    app: tauri::AppHandle,
    state: tauri::State<'_, Mutex<Option<Session>>>,
    request: String,
) -> Result<String, String> {
    let mut guard = state.lock().map_err(|err| err.to_string())?;
    if guard.is_none() {
        let python = chosen_python(&app)?;
        let mut args = vec!["-m".to_string(), "phoenx.gui_session".to_string()];
        let cli: Vec<String> = std::env::args().collect();
        if let Some(pos) = cli.iter().position(|item| item == "--open") {
            if let Some(path) = cli.get(pos + 1) {
                args.push("--open".into());
                args.push(path.clone());
            }
        }
        let mut child = Command::new(&python)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|err| err.to_string())?;
        let stdout = child.stdout.take().ok_or("The session has no output.")?;
        *guard = Some(Session {
            child,
            reader: BufReader::new(stdout),
        });
    }
    let session = guard.as_mut().ok_or("The session is not running.")?;
    let stdin = session.child.stdin.as_mut().ok_or("The session has no input.")?;
    writeln!(stdin, "{request}").map_err(|err| err.to_string())?;
    stdin.flush().map_err(|err| err.to_string())?;
    let mut line = String::new();
    session.reader.read_line(&mut line).map_err(|err| err.to_string())?;
    if line.is_empty() {
        return Err("The session ended.".into());
    }
    serde_json::from_str::<serde_json::Value>(&line).map_err(|_| "not json".to_string())?;
    Ok(line)
}

#[tauri::command]
fn open_folder(path: String) -> Result<(), String> {
    #[cfg(windows)]
    {
        Command::new("explorer")
            .arg(&path)
            .spawn()
            .map_err(|err| err.to_string())?;
    }
    #[cfg(not(windows))]
    {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|err| err.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
    app.exit(0);
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Mutex::new(None::<Session>))
        .invoke_handler(tauri::generate_handler![
            python_status,
            set_python,
            session_call,
            open_folder,
            quit_app
        ])
        .run(tauri::generate_context!())
        .expect("error while running PhoenX RL");
}
