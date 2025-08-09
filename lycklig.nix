{ lib
, stdenv
, cmake
, pkg-config
, opencv4
, tclap
, imagemagick
, boost
, python3
, qt5
}:

let
  pythonPkg = python3.withPackages (ps: with ps; [
    pyqt5
    numpy
    scipy
    opencv-python
  ]);
in stdenv.mkDerivation {
  pname = "lycklig";
  version = "1.0.0";

  src = lib.cleanSource ./.;

  nativeBuildInputs = [
    cmake
    pkg-config
    qt5.wrapQtAppsHook
  ];

  buildInputs = [
    opencv4
    tclap
    imagemagick
    boost
    qt5.qtbase
    pythonPkg
  ];

  # We will wrap kinky only
  dontWrapQtApps = true;

  postFixup = ''
    # Ensure the Python script has proper shebang and can find dependencies
    substituteInPlace $out/bin/kinky \
      --replace "#!/usr/bin/env python3" "#!${pythonPkg}/bin/python3"

    # Ensure that the Qt plugin paths are set
    wrapQtApp "$out/bin/kinky"
  '';

  meta = with lib; {
    description = "Image stacking and correction of atmospheric distortion for lucky imaging";
    longDescription = ''
      lycklig is a part of an astronomical image processing pipeline for
      lucky imaging. It can align and stack a set of images to increase
      signal-to-noise ratio, taking into account image distortions caused by
      the atmosphere.
    '';
    homepage = "https://github.com/AD-Vega/lycklig";
    license = licenses.gpl3Plus;
    maintainers = [ {
      email = "andrej.lajovic@ad-vega.si";
      name = "Andrej Lajovic";
    } {
      email = "jure.varlec@ad-vega.si";
      name = "Jure Varlec";
    } ];
    platforms = platforms.linux;
    mainProgram = "lycklig";
  };
}
