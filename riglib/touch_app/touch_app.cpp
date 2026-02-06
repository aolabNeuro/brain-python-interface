#define WIN32_LEAN_AND_MEAN
#define _WINSOCKK_DEPRICATED_NO_WARNINGS
#define UNICODE
#define _UNICODE
#include <winsock2.h>
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>  // For std::stoi
#include <stdlib.h> // For __argc, __argv

// Link with Ws2_32.lib
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

// --- Globals ---
SOCKET s;
sockaddr_in si_other;
int slen = sizeof(si_other);

// Defaults (can be overridden by command line args)
const char* targetIP = "127.0.0.1";
int targetPort = 5005;

// Touch Flags
#ifndef TOUCHEVENTF_DOWN
#define TOUCHEVENTF_DOWN 0x0002
#endif

// Packet Structure
#pragma pack(push, 1)
struct TouchPacket {
    int id;
    int x;
    int y;
    int flags;
};
#pragma pack(pop)

// Safety Exit Variables
static int tapSequenceCount = 0;
static DWORD lastTapTime = 0;
static HWND hIndicatorWnd = NULL; // Indicator window

// --- Networking Helper ---
void InitUDP(const char* ip, int port) {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    // Set Socket to Non-Blocking Mode
    u_long mode = 1;
    ioctlsocket(s, FIONBIO, &mode);

    memset((char*)&si_other, 0, sizeof(si_other));
    si_other.sin_family = AF_INET;
    si_other.sin_port = htons(port);
    si_other.sin_addr.S_un.S_addr = inet_addr(ip);
}

// --- Cursor Helper ---
// Robustly forces the cursor to show or hide
void SetCursorVisibility(bool show) {
    if (show) {
        while (ShowCursor(TRUE) < 0);
    }
    else {
        while (ShowCursor(FALSE) >= 0);
    }
}

// --- Window Procedure ---
LRESULT CALLBACK IndicatorProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        RECT clientRect;
        GetClientRect(hWnd, &clientRect);
        
        // Fill with green to indicate app is running
        HBRUSH hBrush = CreateSolidBrush(RGB(0, 255, 0));
        FillRect(hdc, &clientRect, hBrush);
        DeleteObject(hBrush);
        
        EndPaint(hWnd, &ps);
        return 0;
    }
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// --- Window Procedure ---
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_TOUCH:
    {
        unsigned int numInputs = LOWORD(wParam);
        std::vector<TOUCHINPUT> inputs(numInputs);
        HTOUCHINPUT hTouchInput = (HTOUCHINPUT)lParam;

        if (GetTouchInputInfo(hTouchInput, numInputs, inputs.data(), sizeof(TOUCHINPUT))) {
            for (const auto& input : inputs) {
                // 1. Packet Construction
                TouchPacket pkt;
                pkt.id = input.dwID;
                pkt.x = input.x / 100; // Touch coords are in 1/100th of a pixel
                pkt.y = input.y / 100;
                pkt.flags = input.dwFlags;

                // 2. Send (Non-blocking)
                sendto(s, (char*)&pkt, sizeof(pkt), 0, (struct sockaddr*)&si_other, slen);

            }
            CloseTouchInputHandle(hTouchInput);
        }
        return 0;
    }

    // Protection: Keep window on top if focus is lost
    case WM_KILLFOCUS:
        SetWindowPos(hWnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
        break;

        // Protection: Block mouse clicks from reaching the underlying VNC window
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP:
    case WM_MOUSEMOVE:
        return 0;

        // Fallback: ESC key still works if a keyboard is attached
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) PostQuitMessage(0);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// --- Main Entry Point ---
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // 1. High-DPI Support (Critical for touch accuracy)
    SetProcessDPIAware();

    // 2. Prevent System Sleep
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);

    // 3. Parse Arguments (IP and Port)
    if (__argc >= 3) {
        targetIP = __argv[1];
        targetPort = std::stoi(__argv[2]);
    }

    InitUDP(targetIP, targetPort);

    // 4. Create Window Class
    const wchar_t* CLASS_NAME = L"TouchOverlayClass";
    WNDCLASSEXW wc = { sizeof(WNDCLASSEXW), CS_HREDRAW | CS_VREDRAW | CS_NOCLOSE, WndProc, 0, 0, hInstance, NULL, NULL, (HBRUSH)GetStockObject(BLACK_BRUSH), NULL, CLASS_NAME, NULL };
    wc.hCursor = NULL; // Default to no cursor
    RegisterClassExW(&wc);

    // 4b. Create Window Class for Indicator
    const wchar_t* INDICATOR_CLASS_NAME = L"IndicatorClass";
    WNDCLASSEXW wcIndicator = { sizeof(WNDCLASSEXW), CS_HREDRAW | CS_VREDRAW, IndicatorProc, 0, 0, hInstance, NULL, NULL, NULL, NULL, INDICATOR_CLASS_NAME, NULL };
    RegisterClassExW(&wcIndicator);

    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    // 5. Create Indicator Window (10x10 pixels in top-right corner)
    int indicatorSize = 10;
    hIndicatorWnd = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        INDICATOR_CLASS_NAME,
        L"Indicator",
        WS_POPUP | WS_VISIBLE,
        screenWidth - indicatorSize, 0, indicatorSize, indicatorSize,
        NULL, NULL, hInstance, NULL
    );

    // 6. Create Overlay Window
    // WS_EX_TRANSPARENT is REMOVED to ensure we BLOCK clicks to the VNC window below.
    HWND hWnd = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TOOLWINDOW,
        CLASS_NAME,
        L"Touch Overlay",
        WS_POPUP | WS_VISIBLE,
        0, 0, screenWidth, screenHeight,
        NULL, NULL, hInstance, NULL
    );

    // 6. Set Transparency (Alpha = 1/255)
    // Almost invisible, but opaque enough to catch input
    SetLayeredWindowAttributes(hWnd, 0, 1, LWA_ALPHA);

    RegisterTouchWindow(hWnd, 0);

    // 7. Hide Cursor
    SetCursorVisibility(false);

    std::cout << "brady's game is working" << std::endl;
    std::cout.flush();

    // 8. Message Loop
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // 9. Cleanup
    if (IsWindow(hIndicatorWnd)) {
        DestroyWindow(hIndicatorWnd);
    }
    SetThreadExecutionState(ES_CONTINUOUS); // Allow sleep
    closesocket(s);
    WSACleanup();
    return 0;
}