use std::{
    net::{TcpStream, ToSocketAddrs},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager,
};
use tauri_plugin_shell::{process::CommandChild, ShellExt};
use uuid::Uuid;

const SIDECAR_NAME: &str = "freerouterd";
const GATEWAY_HOST: &str = "127.0.0.1";
const GATEWAY_PORT: u16 = 8000;

#[derive(Default)]
struct SidecarState {
    child: Arc<Mutex<Option<CommandChild>>>,
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState::default())
        .setup(|app| {
            let token = Uuid::new_v4().to_string();
            let project_root = app
                .path()
                .app_data_dir()
                .map(|path| path.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string());
            let (_rx, child) = app
                .shell()
                .sidecar(SIDECAR_NAME)?
                .args(["--host", GATEWAY_HOST, "--port", &GATEWAY_PORT.to_string()])
                .env("FREEROUTER_DESKTOP_TOKEN", &token)
                .env("FREEROUTER_DESKTOP_PROJECT_ROOT", &project_root)
                .spawn()?;
            *app.state::<SidecarState>().child.lock().unwrap() = Some(child);

            let show = MenuItem::with_id(app, "show", "Show FreeRouter", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;

            TrayIconBuilder::new()
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        if let Some(window) = tray.app_handle().get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                })
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            let handle = app.handle().clone();
            thread::spawn(move || {
                if wait_for_port(GATEWAY_HOST, GATEWAY_PORT, Duration::from_secs(20)) {
                    let url = format!(
                        "http://{}:{}/app-next?desktop_token={}",
                        GATEWAY_HOST, GATEWAY_PORT, token
                    );
                    let window_handle = handle.clone();
                    let _ = handle.run_on_main_thread(move || {
                        if let Some(window) = window_handle.get_webview_window("main") {
                            let _ = window.navigate(url.parse().unwrap());
                        }
                    });
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                if let Some(state) = window.try_state::<SidecarState>() {
                    if let Some(child) = state.child.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running FreeRouter desktop shell");
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
