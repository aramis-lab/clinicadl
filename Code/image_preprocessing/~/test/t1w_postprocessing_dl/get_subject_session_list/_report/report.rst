Node: get_subject_session_list (utility)
========================================

 Hierarchy : t1w_postprocessing_dl.get_subject_session_list
 Exec ID : get_subject_session_list

Original Inputs
---------------

* caps_directory : /network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test
* function_str : def get_caps_t1(caps_directory, tsv):
    """
    THis is a function to grab all the cropped files
    :param caps_directory:
    :param tsv:
    :return:
    """
    import pandas as pd
    import os

    preprocessed_T1 = []
    df = pd.read_csv(tsv, sep='\t')
    if ('session_id' != list(df.columns.values)[1]) and (
                'participant_id' != list(df.columns.values)[0]):
        raise Exception('the data file is not in the correct format.')
    img_list = list(df['participant_id'])
    sess_list = list(df['session_id'])

    for i in range(len(img_list)):
        img_path = os.path.join(caps_directory, 'subjects', img_list[i], sess_list[i], 't1', 'preprocessing_dl', img_list[i] + '_' + sess_list[i] + '_space-MNI_res-1x1x1.nii.gz')
        preprocessed_T1.append(img_path)

    return preprocessed_T1

* ignore_exception : False
* tsv : /teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/test.tsv

Execution Inputs
----------------

* caps_directory : /network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test
* function_str : def get_caps_t1(caps_directory, tsv):
    """
    THis is a function to grab all the cropped files
    :param caps_directory:
    :param tsv:
    :return:
    """
    import pandas as pd
    import os

    preprocessed_T1 = []
    df = pd.read_csv(tsv, sep='\t')
    if ('session_id' != list(df.columns.values)[1]) and (
                'participant_id' != list(df.columns.values)[0]):
        raise Exception('the data file is not in the correct format.')
    img_list = list(df['participant_id'])
    sess_list = list(df['session_id'])

    for i in range(len(img_list)):
        img_path = os.path.join(caps_directory, 'subjects', img_list[i], sess_list[i], 't1', 'preprocessing_dl', img_list[i] + '_' + sess_list[i] + '_space-MNI_res-1x1x1.nii.gz')
        preprocessed_T1.append(img_path)

    return preprocessed_T1

* ignore_exception : False
* tsv : /teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/test.tsv

Execution Outputs
-----------------

* preprocessed_T1 : ['/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test/subjects/sub-ADNI002S0619/ses-M00/t1/preprocessing_dl/sub-ADNI002S0619_ses-M00_space-MNI_res-1x1x1.nii.gz']

Runtime info
------------

* duration : 2.88625
* hostname : UMR-ARAMI-LF011

Environment
~~~~~~~~~~~

* BASH_ENV : /usr/local/Modules/init/bash
* BASH_FUNC_module%% : () {  unset _mlre _mlIFS _mlshdbg;
 if [ "${MODULES_SILENT_SHELL_DEBUG:-0}" = '1' ]; then
 case "$-" in 
 *v*x*)
 set +vx;
 _mlshdbg='vx'
 ;;
 *v*)
 set +v;
 _mlshdbg='v'
 ;;
 *x*)
 set +x;
 _mlshdbg='x'
 ;;
 *)
 _mlshdbg=''
 ;;
 esac;
 fi;
 if [ -n "${IFS+x}" ]; then
 _mlIFS=$IFS;
 fi;
 IFS=' ';
 for _mlv in ${MODULES_RUN_QUARANTINE:-};
 do
 if [ "${_mlv}" = "${_mlv##*[!A-Za-z0-9_]}" -a "${_mlv}" = "${_mlv#[0-9]}" ]; then
 if [ -n "`eval 'echo ${'$_mlv'+x}'`" ]; then
 _mlre="${_mlre:-}${_mlv}_modquar='`eval 'echo ${'$_mlv'}'`' ";
 fi;
 _mlrv="MODULES_RUNENV_${_mlv}";
 _mlre="${_mlre:-}${_mlv}='`eval 'echo ${'$_mlrv':-}'`' ";
 fi;
 done;
 if [ -n "${_mlre:-}" ]; then
 eval `eval ${_mlre}/usr/bin/tclsh /usr/local/Modules/libexec/modulecmd.tcl bash '"$@"'`;
 else
 eval `/usr/bin/tclsh /usr/local/Modules/libexec/modulecmd.tcl bash "$@"`;
 fi;
 _mlstatus=$?;
 if [ -n "${_mlIFS+x}" ]; then
 IFS=$_mlIFS;
 else
 unset IFS;
 fi;
 if [ -n "${_mlshdbg:-}" ]; then
 set -$_mlshdbg;
 fi;
 unset _mlre _mlv _mlrv _mlIFS _mlshdbg;
 return $_mlstatus
}
* BASH_FUNC_switchml%% : () {  typeset swfound=1;
 if [ "${MODULES_USE_COMPAT_VERSION:-0}" = '1' ]; then
 typeset swname='main';
 if [ -e /usr/local/Modules/libexec/modulecmd.tcl ]; then
 typeset swfound=0;
 unset MODULES_USE_COMPAT_VERSION;
 fi;
 else
 typeset swname='compatibility';
 if [ -e /usr/local/Modules/libexec/modulecmd-compat ]; then
 typeset swfound=0;
 MODULES_USE_COMPAT_VERSION=1;
 export MODULES_USE_COMPAT_VERSION;
 fi;
 fi;
 if [ $swfound -eq 0 ]; then
 echo "Switching to Modules $swname version";
 source /usr/local/Modules/init/bash;
 else
 echo "Cannot switch to Modules $swname version, command not found";
 return 1;
 fi
}
* CLUTTER_IM_MODULE : xim
* CMAKE_PREFIX_PATH : /home/junhao.wen/Application/Anaconda2/bin/../
* COLORTERM : gnome-terminal
* COMPIZ_BIN_PATH : /usr/bin/
* COMPIZ_CONFIG_PROFILE : ubuntu
* DBUS_SESSION_BUS_ADDRESS : unix:abstract=/tmp/dbus-dsApnA5dmg
* DEFAULTS_PATH : /usr/share/gconf/ubuntu.default.path
* DEFAULT_USER : wen
* DESKTOP_SESSION : ubuntu
* DISPLAY : :0
* ENV : /usr/local/Modules/init/profile.sh
* FPATH : /usr/local/Modules/init/zsh-functions:/home/junhao.wen/.oh-my-zsh/plugins/git:/home/junhao.wen/.oh-my-zsh/functions:/home/junhao.wen/.oh-my-zsh/completions:/usr/local/share/zsh/site-functions:/usr/share/zsh/vendor-functions:/usr/share/zsh/vendor-completions:/usr/share/zsh/functions/Calendar:/usr/share/zsh/functions/Chpwd:/usr/share/zsh/functions/Completion:/usr/share/zsh/functions/Completion/AIX:/usr/share/zsh/functions/Completion/BSD:/usr/share/zsh/functions/Completion/Base:/usr/share/zsh/functions/Completion/Cygwin:/usr/share/zsh/functions/Completion/Darwin:/usr/share/zsh/functions/Completion/Debian:/usr/share/zsh/functions/Completion/Linux:/usr/share/zsh/functions/Completion/Mandriva:/usr/share/zsh/functions/Completion/Redhat:/usr/share/zsh/functions/Completion/Solaris:/usr/share/zsh/functions/Completion/Unix:/usr/share/zsh/functions/Completion/X:/usr/share/zsh/functions/Completion/Zsh:/usr/share/zsh/functions/Completion/openSUSE:/usr/share/zsh/functions/Exceptions:/usr/share/zsh/functions/MIME:/usr/share/zsh/functions/Misc:/usr/share/zsh/functions/Newuser:/usr/share/zsh/functions/Prompts:/usr/share/zsh/functions/TCP:/usr/share/zsh/functions/VCS_Info:/usr/share/zsh/functions/VCS_Info/Backends:/usr/share/zsh/functions/Zftp:/usr/share/zsh/functions/Zle
* GDMSESSION : ubuntu
* GDM_LANG : fr_FR
* GNOME_DESKTOP_SESSION_ID : this-is-deprecated
* GNOME_KEYRING_CONTROL : /run/user/17395/keyring-zqbO7P
* GNOME_KEYRING_PID : 2098
* GPG_AGENT_INFO : /run/user/17395/keyring-zqbO7P/gpg:0:1
* GTK_IM_MODULE : ibus
* GTK_MODULES : overlay-scrollbar:unity-gtk-module
* HOME : /home/junhao.wen
* IM_CONFIG_PHASE : 1
* INSTANCE : 
* JOB : dbus
* KRB5CCNAME : FILE:/tmp/krb5cc_17395_SXUXNE
* LANG : en_US.UTF-8
* LANGUAGE : fr_FR
* LC_ALL : en_US.UTF-8
* LC_CTYPE : fr_FR.UTF-8
* LD_LIBRARY_PATH : /home/junhao.wen/Application/pycharm-community-2017.2.3/bin:/usr/local/cuda-9.2/lib64:
* LESS : -R
* LOADEDMODULES : 
* LOGNAME : junhao.wen
* LSCOLORS : Gxfxcxdxbxegedabagacad
* LS_COLORS : rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arj=01;31:*.taz=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lz=01;31:*.xz=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.jpg=01;35:*.jpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.axv=01;35:*.anx=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.axa=00;36:*.oga=00;36:*.spx=00;36:*.xspf=00;36:
* MANDATORY_PATH : /usr/share/gconf/ubuntu.mandatory.path
* MKL_THREADING_LAYER : GNU
* MODULEPATH : /network/lustre/iss01/apps/teams/aramis/clinica/modulefiles:/usr/local/Modules/modulefiles
* MODULEPATH_modshare : /network/lustre/iss01/apps/teams/aramis/clinica/modulefiles:1:/usr/local/Modules/modulefiles:1
* MODULESHOME : /usr/local/Modules
* MODULES_CMD : /usr/local/Modules/libexec/modulecmd.tcl
* NIFTI_MATLIB_TOOLBOX : /home/junhao.wen/Application/Niftimatlib/niftimatlib-1.2
* NODDI_MATLAB_TOOLBOX : /home/junhao.wen/Application/NODDI_matlab_toolbox/NODDI_toolbox_v1.01
* OLDPWD : /home/junhao.wen/Application/pycharm-community-2017.2.3/bin
* PAGER : less
* PATH : /home/junhao.wen/Application/Anaconda2/bin:/home/junhao.wen/Application/Anaconda2/condabin:/usr/local/Modules/bin:/usr/local/cuda-9.2/bin:/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Code/image_preprocessing/optiBET:/home/junhao.wen/Application/Anaconda2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
* PWD : /teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Code/image_preprocessing
* PYCHARM_HOSTED : 1
* PYTHONIOENCODING : UTF-8
* PYTHONPATH : /teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL
* PYTHONUNBUFFERED : 1
* QT4_IM_MODULE : ibus
* QT_IM_MODULE : ibus
* QT_QPA_PLATFORMTHEME : appmenu-qt5
* SELINUX_INIT : YES
* SESSION : ubuntu
* SESSIONTYPE : gnome-session
* SESSION_MANAGER : local/UMR-ARAMI-LF011:@/tmp/.ICE-unix/2308,unix/UMR-ARAMI-LF011:/tmp/.ICE-unix/2308
* SHELL : /bin/zsh
* SHLVL : 1
* SSH_AUTH_SOCK : /run/user/17395/keyring-zqbO7P/ssh
* TERM : xterm
* TEXTDOMAIN : im-config
* TEXTDOMAINDIR : /usr/share/locale/
* UPSTART_SESSION : unix:abstract=/com/ubuntu/upstart-session/17395/2100
* USER : junhao.wen
* VTE_VERSION : 3409
* WINDOWID : 16777227
* XAUTHORITY : /home/junhao.wen/.Xauthority
* XDG_CONFIG_DIRS : /etc/xdg/xdg-ubuntu:/usr/share/upstart/xdg:/etc/xdg
* XDG_CURRENT_DESKTOP : Unity
* XDG_DATA_DIRS : /usr/share/ubuntu:/usr/share/gnome:/usr/local/share/:/usr/share/
* XDG_GREETER_DATA_DIR : /var/lib/lightdm-data/junhao.wen
* XDG_MENU_PREFIX : gnome-
* XDG_RUNTIME_DIR : /run/user/17395
* XDG_SEAT : seat0
* XDG_SEAT_PATH : /org/freedesktop/DisplayManager/Seat0
* XDG_SESSION_ID : c2
* XDG_SESSION_PATH : /org/freedesktop/DisplayManager/Session0
* XDG_VTNR : 7
* XMODIFIERS : @im=ibus
* ZSH : /home/junhao.wen/.oh-my-zsh
* _ : /home/junhao.wen/Application/pycharm-community-2017.2.3/bin/./pycharm.sh

