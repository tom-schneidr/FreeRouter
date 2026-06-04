import React from "react";
import { applyTheme, getThemePreference } from "../theme";

type EmbedFrameProps = {
  title: string;
  src: string;
};

export function EmbedFrame({ title, src }: EmbedFrameProps) {
  const iframeRef = React.useRef<HTMLIFrameElement>(null);

  React.useEffect(() => {
    const frame = iframeRef.current;
    if (!frame) return;
    const onLoad = () => {
      applyTheme(getThemePreference(), { persist: false, broadcast: true });
    };
    frame.addEventListener("load", onLoad);
    return () => frame.removeEventListener("load", onLoad);
  }, [src]);

  return (
    <iframe
      ref={iframeRef}
      className="embed-frame"
      title={title}
      src={src}
      loading="lazy"
    />
  );
}
