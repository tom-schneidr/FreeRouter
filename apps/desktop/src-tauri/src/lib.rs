use std::{
    fs::{create_dir_all, OpenOptions},
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager, RunEvent, WindowEvent,
};
use tauri_plugin_shell::{process::CommandEvent, process::CommandChild, ShellExt};
use uuid::Uuid;

const SIDECAR_NAME: &str = "freerouterd";
const GATEWAY_HOST: &str = "127.0.0.1";
const DEFAULT_GATEWAY_PORT: u16 = 8000;
const SIDECAR_RESTART_EXIT_CODE: i32 = 42;
const DESKTOP_APP_ROUTE: &str = "/app";
const TRAY_ICON_ID: &str = "main";
const GATEWAY_PORT_SWEEP: u16 = 20;

struct AppRuntimeState {
    sidecar: Mutex<Option<CommandChild>>,
    gateway_port: u16,
    desktop_token: String,
    project_root: PathBuf,
    manages_sidecar: bool,
    quitting: AtomicBool,
    #[cfg(windows)]
    sidecar_job: Mutex<Option<isize>>,
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let dev_backend = std::env::var("FREEROUTER_DEV_BACKEND").ok().as_deref() == Some("1");
            let token = std::env::var("FREEROUTER_DESKTOP_TOKEN")
                .unwrap_or_else(|_| Uuid::new_v4().to_string());
            let gateway_port = configured_gateway_port();
            let project_root = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| PathBuf::from("."));

            app.manage(AppRuntimeState {
                sidecar: Mutex::new(None),
                gateway_port,
                desktop_token: token.clone(),
                project_root: project_root.clone(),
                manages_sidecar: !dev_backend,
                quitting: AtomicBool::new(false),
                #[cfg(windows)]
                sidecar_job: Mutex::new(None),
            });

            if dev_backend {
                let _ = append_launcher_log(
                    &project_root,
                    &format!(
                        "Using existing FreeRouter dev backend on {GATEWAY_HOST}:{gateway_port}."
                    ),
                );
            } else {
                reclaim_gateway_port(GATEWAY_HOST, gateway_port);
                start_sidecar(app.handle(), gateway_port, &token, &project_root);
            }

            let show = MenuItem::with_id(app, "show", "Show FreeRouter", true, None::<&str>)?;
            let hide_to_tray =
                MenuItem::with_id(app, "hide_to_tray", "Hide to tray", true, None::<&str>)?;
            let chat = MenuItem::with_id(app, "open_chat", "Chat", true, None::<&str>)?;
            let models = MenuItem::with_id(app, "open_models", "Models", true, None::<&str>)?;
            let copy_url =
                MenuItem::with_id(app, "copy_url", "Copy base URL", true, None::<&str>)?;
            let restart =
                MenuItem::with_id(app, "restart", "Restart server", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(
                app,
                &[&show, &hide_to_tray, &chat, &models, &copy_url, &restart, &quit],
            )?;

            let tray_icon = app
                .default_window_icon()
                .cloned()
                .expect("Missing application icon");

            TrayIconBuilder::with_id(TRAY_ICON_ID)
                .icon(tray_icon)
                .tooltip("FreeRouter")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main_window(tray.app_handle());
                    }
                })
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => show_main_window(app),
                    "hide_to_tray" => hide_main_window(app),
                    "open_chat" => navigate_desktop_section(app, "chat"),
                    "open_models" => navigate_desktop_section(app, "models"),
                    "copy_url" => copy_base_url(app),
                    "restart" => {
                        restart_sidecar(app);
                    }
                    "quit" => quit_application(app),
                    _ => {}
                })
                .build(app)?;

            let handle = app.handle().clone();
            thread::spawn(move || {
                if wait_for_port(GATEWAY_HOST, gateway_port, Duration::from_secs(20)) {
                    let url = desktop_app_url(gateway_port, &token, None);
                    let window_handle = handle.clone();
                    let _ = handle.run_on_main_thread(move || {
                        if let Some(window) = window_handle.get_webview_window("main") {
                            let _ = window.navigate(url.parse().unwrap());
                        }
                    });
                }
            });

            spawn_gateway_health_watchdog(app.handle().clone());

            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                hide_main_window(window.app_handle());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building FreeRouter desktop shell")
        .run(|app_handle, event| {
            match event {
                RunEvent::Exit | RunEvent::ExitRequested { .. } => {
                    shutdown_gateway(app_handle);
                }
                _ => {}
            }
        });
}

fn start_sidecar(
    app: &AppHandle,
    gateway_port: u16,
    token: &str,
    project_root: &PathBuf,
) {
    let _ = append_launcher_log(
        project_root,
        &format!("Starting FreeRouter sidecar on {GATEWAY_HOST}:{gateway_port}."),
    );

    let gateway_port_arg = gateway_port.to_string();
    let (rx, child) = app
        .shell()
        .sidecar(SIDECAR_NAME)
        .expect("Failed to create sidecar command")
        .args(["--host", GATEWAY_HOST, "--port", gateway_port_arg.as_str()])
        .env("FREEROUTER_DESKTOP_TOKEN", token)
        .env(
            "FREEROUTER_APP_DATA_DIR",
            project_root.to_string_lossy().as_ref(),
        )
        .env(
            "FREEROUTER_DESKTOP_PROJECT_ROOT",
            project_root.to_string_lossy().as_ref(),
        )
        .spawn()
        .expect("Failed to spawn FreeRouter sidecar");

    let sidecar_pid = child.pid();
    if let Some(state) = app.try_state::<AppRuntimeState>() {
        *state.sidecar.lock().unwrap() = Some(child);
        #[cfg(windows)]
        if let Some(job) = create_kill_on_close_job(sidecar_pid) {
            *state.sidecar_job.lock().unwrap() = Some(job);
        }
    }

    spawn_sidecar_monitor(app.clone(), rx, gateway_port, token.to_string(), project_root.clone());
}

fn spawn_sidecar_monitor(
    app: AppHandle,
    mut rx: tauri::async_runtime::Receiver<CommandEvent>,
    gateway_port: u16,
    token: String,
    project_root: PathBuf,
) {
    thread::spawn(move || {
        while let Some(event) = tauri::async_runtime::block_on(rx.recv()) {
            match event {
                CommandEvent::Stdout(line) | CommandEvent::Stderr(line) => {
                    append_sidecar_output(&project_root, &line);
                }
                CommandEvent::Terminated(payload) => {
                    let quitting = app
                        .try_state::<AppRuntimeState>()
                        .map(|state| state.quitting.load(Ordering::SeqCst))
                        .unwrap_or(false);
                    if quitting {
                        break;
                    }
                    if payload.code == Some(SIDECAR_RESTART_EXIT_CODE) {
                        let _ = append_launcher_log(
                            &project_root,
                            "Desktop restart requested by local app controls.",
                        );
                    } else {
                        let _ = append_launcher_log(
                            &project_root,
                            &format!(
                                "FreeRouter sidecar exited unexpectedly with code {:?}; restarting.",
                                payload.code
                            ),
                        );
                    }
                    thread::sleep(Duration::from_millis(500));
                    reclaim_gateway_port(GATEWAY_HOST, gateway_port);
                    start_sidecar(&app, gateway_port, &token, &project_root);
                    break;
                }
                CommandEvent::Error(message) => {
                    let _ = append_launcher_log(&project_root, &format!("Sidecar error: {message}"));
                }
                _ => {}
            }
        }
    });
}

fn restart_sidecar(app: &AppHandle) {
    let Some(state) = app.try_state::<AppRuntimeState>() else {
        return;
    };

    if !state.manages_sidecar {
        let _ = append_launcher_log(
            &state.project_root,
            "Restart requested in dev backend mode; restart the dev script instead.",
        );
        return;
    }

    let gateway_port = state.gateway_port;
    let token = state.desktop_token.clone();
    let project_root = state.project_root.clone();

    shutdown_sidecar(app);
    thread::sleep(Duration::from_millis(350));
    reclaim_gateway_port(GATEWAY_HOST, gateway_port);
    start_sidecar(app, gateway_port, &token, &project_root);
}

fn shutdown_sidecar(app: &AppHandle) {
    if let Some(state) = app.try_state::<AppRuntimeState>() {
        if let Some(child) = state.sidecar.lock().unwrap().take() {
            let pid = child.pid();
            let _ = child.kill();
            force_kill_process_tree(pid);
        }
        #[cfg(windows)]
        if let Some(job) = state.sidecar_job.lock().unwrap().take() {
            close_job_handle(job);
        }
    }
}

fn shutdown_gateway(app: &AppHandle) {
    let gateway_port = app
        .try_state::<AppRuntimeState>()
        .map(|state| state.gateway_port)
        .unwrap_or(DEFAULT_GATEWAY_PORT);
    shutdown_sidecar(app);
    kill_dev_backend_process_tree();
    for offset in 0..=GATEWAY_PORT_SWEEP {
        stop_gateway_listeners(GATEWAY_HOST, gateway_port.saturating_add(offset));
    }
}

fn kill_dev_backend_process_tree() {
    let Ok(pid_raw) = std::env::var("FREEROUTER_DEV_BACKEND_PID") else {
        return;
    };
    let Ok(pid) = pid_raw.parse::<u32>() else {
        return;
    };
    if pid > 0 {
        force_kill_process_tree(pid);
    }
}

fn quit_application(app: &AppHandle) {
    if let Some(state) = app.try_state::<AppRuntimeState>() {
        if state.quitting.swap(true, Ordering::SeqCst) {
            return;
        }
    }
    remove_tray_icon(app);
    shutdown_gateway(app);
    // Brief pause so taskkill / job teardown can finish before the shell exits.
    thread::sleep(Duration::from_millis(450));
    shutdown_gateway(app);
    app.exit(0);
}

fn remove_tray_icon(app: &AppHandle) {
    if let Some(tray) = app.tray_by_id(TRAY_ICON_ID) {
        let _ = tray.set_visible(false);
    }
    let _ = app.remove_tray_by_id(TRAY_ICON_ID);
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn hide_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

fn navigate_desktop_section(app: &AppHandle, section: &str) {
    let Some(state) = app.try_state::<AppRuntimeState>() else {
        return;
    };
    let url = desktop_app_url(state.gateway_port, &state.desktop_token, Some(section));
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.navigate(url.parse().unwrap());
        let _ = window.set_focus();
    }
}

fn copy_base_url(app: &AppHandle) {
    let Some(state) = app.try_state::<AppRuntimeState>() else {
        return;
    };
    let base_url = format!("http://{}:{}/v1", GATEWAY_HOST, state.gateway_port);
    copy_text_to_clipboard(&base_url);
}

fn desktop_app_url(port: u16, token: &str, section: Option<&str>) -> String {
    let mut url = format!(
        "http://{}:{}{DESKTOP_APP_ROUTE}?desktop_token={token}",
        GATEWAY_HOST, port
    );
    if let Some(section) = section {
        url.push('#');
        url.push_str(section);
    }
    url
}

fn copy_text_to_clipboard(text: &str) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;

        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let escaped = text.replace('\'', "''");
        let _ = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("Set-Clipboard -Value '{escaped}'"),
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .status();
    }

    #[cfg(not(windows))]
    {
        let _ = text;
    }
}

fn append_sidecar_output(project_root: &PathBuf, bytes: &[u8]) {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return;
    };
    if text.is_empty() {
        return;
    }
    let _ = append_launcher_log(project_root, text.trim_end());
}

fn append_launcher_log(project_root: &PathBuf, message: &str) -> std::io::Result<()> {
    let log_dir = project_root.join("data");
    create_dir_all(&log_dir)?;
    let log_path = log_dir.join("desktop-app.log");
    let mut file = OpenOptions::new().create(true).append(true).open(log_path)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    for line in message.lines() {
        writeln!(file, "[{timestamp}] {line}")?;
    }
    Ok(())
}

#[cfg(windows)]
fn force_kill_process_tree(pid: u32) {
    use std::os::windows::process::CommandExt;

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .creation_flags(CREATE_NO_WINDOW)
        .status();
}

#[cfg(not(windows))]
fn force_kill_process_tree(_pid: u32) {}

#[cfg(windows)]
fn close_job_handle(job: isize) {
    use windows::Win32::Foundation::{CloseHandle, HANDLE};

    if job != 0 {
        unsafe {
            let _ = CloseHandle(HANDLE(job as *mut _));
        }
    }
}

#[cfg(windows)]
fn create_kill_on_close_job(pid: u32) -> Option<isize> {
    use std::mem::size_of;

    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE};

    if pid == 0 {
        return None;
    }

    unsafe {
        let job = CreateJobObjectW(None, None).ok()?;

        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        windows::Win32::System::JobObjects::SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
        .ok()?;

        let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, false, pid).ok()?;

        if AssignProcessToJobObject(job, process).is_err() {
            let _ = CloseHandle(process);
            let _ = CloseHandle(job);
            return None;
        }

        let _ = CloseHandle(process);
        Some(job.0 as isize)
    }
}

fn configured_gateway_port() -> u16 {
    std::env::var("GATEWAY_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(DEFAULT_GATEWAY_PORT)
}

fn reclaim_gateway_port(host: &str, port: u16) {
    stop_gateway_listeners(host, port);
    thread::sleep(Duration::from_millis(350));
}

fn listener_pids_on_port(host: &str, port: u16) -> Vec<u32> {
    let mut pids = Vec::new();
    let needle = format!("{host}:{port}");

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;

        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let output = match std::process::Command::new("netstat")
            .args(["-ano"])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
        {
            Ok(output) => output,
            Err(_) => return pids,
        };

        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines() {
            let trimmed = line.trim();
            if !trimmed.contains("LISTENING") || !trimmed.contains(&needle) {
                continue;
            }
            let Some(pid) = trimmed.split_whitespace().last().and_then(|value| value.parse::<u32>().ok())
            else {
                continue;
            };
            if pid > 0 && !pids.contains(&pid) {
                pids.push(pid);
            }
        }
    }

    #[cfg(not(windows))]
    {
        let _ = (host, port, needle);
    }

    pids
}

fn stop_gateway_listeners(host: &str, port: u16) {
    for pid in listener_pids_on_port(host, port) {
        force_kill_process_tree(pid);
    }
    thread::sleep(Duration::from_millis(120));
    for pid in listener_pids_on_port(host, port) {
        force_kill_process_tree(pid);
    }
}

fn spawn_gateway_health_watchdog(app: AppHandle) {
    thread::spawn(move || {
        let mut consecutive_failures = 0u32;
        loop {
            thread::sleep(Duration::from_secs(30));
            let Some(state) = app.try_state::<AppRuntimeState>() else {
                break;
            };
            if state.quitting.load(Ordering::SeqCst) {
                break;
            }
            let port = state.gateway_port;
            if gateway_health_probe(GATEWAY_HOST, port) {
                consecutive_failures = 0;
                continue;
            }
            consecutive_failures += 1;
            if consecutive_failures < 3 {
                continue;
            }
            let project_root = state.project_root.clone();
            let manages_sidecar = state.manages_sidecar;
            let _ = append_launcher_log(
                &project_root,
                &format!(
                    "Gateway health check failed on {GATEWAY_HOST}:{port}; attempting recovery."
                ),
            );
            consecutive_failures = 0;
            if manages_sidecar {
                restart_sidecar(&app);
            }
        }
    });
}

fn gateway_health_probe(host: &str, port: u16) -> bool {
    let address = match (host, port).to_socket_addrs() {
        Ok(mut addresses) => addresses.next(),
        Err(_) => None,
    };
    let Some(address) = address else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_secs(2)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request = format!(
        "GET /v1/gateway/health.json HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut buffer = [0_u8; 768];
    let Ok(read) = stream.read(&mut buffer) else {
        return false;
    };
    let response = std::str::from_utf8(&buffer[..read]).unwrap_or("");
    response.contains("200 OK") && response.contains("\"freerouter\"")
}

fn wait_for_port(host: &str, port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let address = (host, port)
            .to_socket_addrs()
            .ok()
            .and_then(|mut addresses| addresses.next());
        if let Some(address) = address {
            if TcpStream::connect_timeout(&address, Duration::from_millis(350)).is_ok() {
                return true;
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    false
}
