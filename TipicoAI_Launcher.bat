@echo off
title TRELLIS / TipicoAI Launcher
color 0A

setlocal EnableExtensions EnableDelayedExpansion

REM Always run relative to this launcher location
pushd "%~dp0" >nul
if errorlevel 1 (
    echo Error: Failed to set working directory to: %~dp0
    pause
    exit /b 1
)

REM ====== Config ======
set "REPO_URL=https://github.com/EpicGamesVerse/TrellisAI"
set "REPO_DIR=trellisai"
set "VENV_ACT=%REPO_DIR%\venv\Scripts\activate.bat"
set "MODELS_DIR=%~dp0%REPO_DIR%\models"
set "MODEL_REPO_ID=MonsterMMORPG/SECourses_Rock"

REM Make console output stable (prevents garbled tqdm progress output)
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "TQDM_DISABLE=1"

:menu
cls
echo ====================================
echo      TRELLIS / TipicoAI Launcher
echo ====================================
echo.
echo 1) Start TRELLIS (Normal GPU)
echo 2) Start TRELLIS (Low VRAM / FP16)
echo 3) Download / Update Models
echo 4) Install / Repair TRELLIS
echo 5) RunPod Install (WSL)
echo 6) Update TRELLIS
echo 7) Uninstall Flash Attention (RTX 2000 ^& below)
echo 8) Massed Compute Install (WSL)
echo 9) Exit
echo.

set "choice="
set /p "choice=Choose an option (1-9): "

if "%choice%"=="1" goto start_normal
if "%choice%"=="2" goto start_low
if "%choice%"=="3" goto models
if "%choice%"=="4" goto install
if "%choice%"=="5" goto runpod_install
if "%choice%"=="6" goto update
if "%choice%"=="7" goto uninstall_flashattn
if "%choice%"=="8" goto massed_install
if "%choice%"=="9" goto end

echo.
echo Invalid choice: "%choice%"
pause
goto menu

REM =========================
REM Menu actions
REM =========================

:start_normal
cls
echo === Start TRELLIS (Normal GPU) ===
call :ensure_repo || goto failback
call :activate_venv || goto failback

set "PYTHONWARNINGS=ignore"
set "HF_HUB_ENABLE_HF_TRANSFER=1"

set "TRANSFORMERS_CACHE=%MODELS_DIR%"
set "HF_HOME=%MODELS_DIR%"
set "HF_DATASETS_CACHE=%MODELS_DIR%"
echo Hugging Face cache folders set to: %TRANSFORMERS_CACHE%
echo.

pushd "%REPO_DIR%" >nul || goto failback
python tipico_trellis.py --highvram
set "rc=%ERRORLEVEL%"
popd >nul

if not "%rc%"=="0" (
    echo.
    echo Error: TRELLIS exited with errorlevel=%rc%
    pause
)
goto menu

:start_low
cls
echo === Start TRELLIS (Low VRAM / FP16) ===
call :ensure_repo || goto failback
call :activate_venv || goto failback

set "PYTHONWARNINGS=ignore"
set "HF_HUB_ENABLE_HF_TRANSFER=1"

set "TRANSFORMERS_CACHE=%MODELS_DIR%"
set "HF_HOME=%MODELS_DIR%"
set "HF_DATASETS_CACHE=%MODELS_DIR%"
echo Hugging Face cache folders set to: %TRANSFORMERS_CACHE%
echo.

pushd "%REPO_DIR%" >nul || goto failback
python tipico_trellis.py --precision fp16
set "rc=%ERRORLEVEL%"
popd >nul

if not "%rc%"=="0" (
    echo.
    echo Error: TRELLIS exited with errorlevel=%rc%
    pause
)
goto menu

:models
cls
echo === Download / Update Models ===
call :ensure_repo || goto failback
call :activate_venv_optional
call :download_models || goto failback
echo.
echo DONE: Model download completed.
pause
goto menu

:install
cls
echo === Install / Repair TRELLIS ===
call :ensure_tools || goto failback
call :ensure_repo || goto failback
call :ensure_venv || goto failback
call :activate_venv || goto failback
call :pip_install_all || goto failback
call :download_models || goto failback
echo.
echo DONE: Virtual environment made and installed properly.
pause
goto menu

:update
cls
echo === Update TRELLIS ===
call :ensure_tools || goto failback
call :ensure_repo || goto failback

pushd "%REPO_DIR%" >nul || goto failback
git pull
popd >nul

call :ensure_venv || goto failback
call :activate_venv || goto failback
call :pip_install_all || goto failback
call :download_models || goto failback
echo.
echo DONE: Update completed.
pause
goto menu

:uninstall_flashattn
cls
echo === Uninstall Flash Attention (RTX 2000 ^& below) ===
call :ensure_repo || goto failback
call :activate_venv || goto failback

pushd "%REPO_DIR%" >nul || goto failback
python -m pip uninstall flash_attn --yes
set "rc=%ERRORLEVEL%"
popd >nul

echo.
if "%rc%"=="0" (
    echo flash-attn uninstalled.
) else (
    echo flash-attn uninstall finished with errorlevel=%rc% (may already be uninstalled).
)
pause
goto menu

:runpod_install
cls
echo === RunPod Install (WSL) ===
echo This runs the RunPod install script inside WSL.
echo.

set "TMP_SH=%TEMP%\tipicoai_runpod_install_%RANDOM%%RANDOM%.sh"
(
    echo apt-get update --yes
    echo apt-get install python3-tk --yes
    echo apt update --yes
    echo apt install ninja-build --yes
    echo pip install requests
    echo pip install tqdm
    echo git lfs install
    echo.
    echo cd /workspace
    echo.
    echo git clone --recursive https://github.com/EpicGamesVerse/TrellisAI trellisai
    echo.
    echo cd trellisai
    echo.
    echo python -m venv venv
    echo.
    echo source ./venv/bin/activate
    echo.
    echo python -m pip install --upgrade pip
    echo.
    echo pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo.
    echo pip install gradio gradio_litmodel3d
    echo.
    echo pip install pydantic==2.10.6
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/xformers-0.0.30+3abeaa9e.d20250427-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install deepspeed
    echo.
    echo pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers torchao natsort
    echo.
    echo pip install spconv-cu120
    echo.
    echo pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    echo.
    echo pip install huggingface_hub ipywidgets hf_transfer hf_xet
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/diffoctreerast-0.0.0-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/nvdiffrast-0.3.3-py3-none-any.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/kaolin-0.17.0-cp310-cp310-linux_x86_64.whl
    echo.
    echo export HF_HUB_ENABLE_HF_TRANSFER=1
    echo.
    echo cd ..
    echo.
    echo echo "RunPod install completed."
    echo read -p "Press Enter to continue"
) > "%TMP_SH%"

call :run_wsl_file "%TMP_SH%"
set "WSL_RC=%ERRORLEVEL%"
del /f /q "%TMP_SH%" >nul 2>&1

if not "%WSL_RC%"=="0" (
    echo.
    echo RunPod install failed.
    pause
    goto menu
)

echo.
echo RunPod install finished (check output above for errors).
pause
goto menu

:massed_install
cls
echo === Massed Compute Install (WSL) ===
echo This runs the Massed Compute install script inside WSL.
echo.

set "TMP_SH=%TEMP%\tipicoai_massed_install_%RANDOM%%RANDOM%.sh"
(
    echo pip install requests
    echo pip install tqdm
    echo sudo apt update
    echo git lfs install
    echo sudo apt update --yes
    echo.
    echo git clone --recursive https://github.com/EpicGamesVerse/TrellisAI trellisai
    echo.
    echo cd trellisai
    echo.
    echo python3 -m venv venv
    echo.
    echo source ./venv/bin/activate
    echo.
    echo python3 -m pip install --upgrade pip
    echo.
    echo pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo.
    echo pip install gradio gradio_litmodel3d
    echo.
    echo pip install pydantic==2.10.6
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/xformers-0.0.30+3abeaa9e.d20250427-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install deepspeed
    echo.
    echo pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers torchao natsort
    echo.
    echo pip install spconv-cu120
    echo.
    echo pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    echo.
    echo pip install huggingface_hub ipywidgets hf_transfer hf_xet
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/diffoctreerast-0.0.0-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/nvdiffrast-0.3.3-py3-none-any.whl
    echo.
    echo pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/kaolin-0.17.0-cp310-cp310-linux_x86_64.whl
    echo.
    echo export HF_HUB_ENABLE_HF_TRANSFER=1
    echo.
    echo cd ..
    echo.
    echo echo "Massed Compute install completed."
    echo read -p "Press Enter to continue"
) > "%TMP_SH%"

call :run_wsl_file "%TMP_SH%"
set "WSL_RC=%ERRORLEVEL%"
del /f /q "%TMP_SH%" >nul 2>&1

if not "%WSL_RC%"=="0" (
    echo.
    echo Massed Compute install failed.
    pause
    goto menu
)

echo.
echo Massed Compute install finished (check output above for errors).
pause
goto menu

REM =========================
REM Helpers
REM =========================

:ensure_tools
where git >nul 2>&1 || (echo Error: git not found in PATH. & exit /b 1)
where py >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1 || (echo Error: python not found in PATH. & exit /b 1)
)
exit /b 0

:ensure_repo
if exist "%REPO_DIR%\." exit /b 0
git lfs install >nul 2>&1
echo Cloning TRELLIS...
git clone --recursive "%REPO_URL%" "%REPO_DIR%"
exit /b %ERRORLEVEL%

:ensure_venv
if exist "%VENV_ACT%" exit /b 0

echo Creating venv...
pushd "%REPO_DIR%" >nul || exit /b 1

py --version >nul 2>&1
if "%ERRORLEVEL%"=="0" (
    echo Python launcher detected. Creating Python 3.10 venv...
    py -3.10 -m venv venv
) else (
    echo Python launcher not detected. Creating venv with default python...
    python -m venv venv
)

set "rc=%ERRORLEVEL%"
popd >nul
exit /b %rc%

:activate_venv
if not exist "%VENV_ACT%" (
    echo Error: venv not found. Run Install first.
    exit /b 1
)
call "%VENV_ACT%"
exit /b %ERRORLEVEL%

:activate_venv_optional
if exist "%VENV_ACT%" (
    call "%VENV_ACT%"
) else (
    echo Warning: venv not found. Using system Python/Pip for model download dependencies.
)
exit /b 0

:pip_install_all
echo Upgrading pip...
python -m pip install --upgrade pip || exit /b 1

echo Installing/Updating requirements...
python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
python -m pip install gradio gradio_litmodel3d || exit /b 1
python -m pip install pydantic==2.10.6 || exit /b 1

python -m pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl || exit /b 1
python -m pip install https://files.pythonhosted.org/packages/15/b0/be6cc74fd1e23da20d6c34db923858a8ae5017d39a13dedc188a935c646a/deepspeed-0.16.5-cp310-cp310-win_amd64.whl || exit /b 1
python -m pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu128torch2.7.0-cp310-cp310-win_amd64.whl || exit /b 1
python -m pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/xformers-0.0.30+3abeaa9e.d20250424-cp310-cp310-win_amd64.whl || exit /b 1
python -m pip install triton-windows==3.3.0.post19 --upgrade || exit /b 1

python -m pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers torchao natsort || exit /b 1
python -m pip install spconv-cu120 || exit /b 1
python -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 || exit /b 1
python -m pip install huggingface_hub ipywidgets hf_transfer hf_xet || exit /b 1

python -m pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/diffoctreerast-0.0.0-cp310-cp310-win_amd64.whl || exit /b 1
python -m pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/diff_gaussian_rasterization-0.0.0-cp310-cp310-win_amd64.whl || exit /b 1
python -m pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/nvdiffrast-0.3.3-py3-none-any.whl || exit /b 1
python -m pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/kaolin-0.17.0-cp310-cp310-win_amd64.whl || exit /b 1

exit /b 0

:download_models
echo Downloading models...
set "HF_HUB_ENABLE_HF_TRANSFER=1"
set "TRANSFORMERS_CACHE=%MODELS_DIR%"
set "HF_HOME=%MODELS_DIR%"
set "HF_DATASETS_CACHE=%MODELS_DIR%"
python -m pip install huggingface_hub ipywidgets hf_transfer hf_xet || exit /b 1
python -c "from huggingface_hub import snapshot_download; repo_id=r'%MODEL_REPO_ID%'; local_dir=r'%MODELS_DIR%'; snapshot_download(repo_id=repo_id, local_dir=local_dir); print('.\n.\nDOWNLOAD COMPLETED')"
exit /b %ERRORLEVEL%

:run_wsl_file
set "WIN_FILE=%~1"

where wsl >nul 2>&1
if errorlevel 1 (
    echo Error: WSL not found on this Windows install.
    echo Install WSL or run this step on a Linux machine.
    exit /b 1
)

if not exist "%WIN_FILE%" (
    echo Error: File not found: %WIN_FILE%
    exit /b 1
)

for /f "usebackq delims=" %%P in (`wsl wslpath -a "%WIN_FILE%" 2^>nul`) do set "WSL_FILE=%%P"
if not defined WSL_FILE (
    echo Error: Failed to resolve WSL path for %WIN_FILE%
    exit /b 1
)

echo Starting WSL script...
echo (You may be prompted for sudo inside WSL.)
echo.

wsl bash -lc "set -e; sed -i 's/\r$//' \"%WSL_FILE%\"; chmod +x \"%WSL_FILE%\"; \"%WSL_FILE%\""
exit /b %ERRORLEVEL%

:failback
echo.
echo FAILED. Check the messages above for the first error.
pause
goto menu

:end
popd >nul
endlocal
exit /b 0
