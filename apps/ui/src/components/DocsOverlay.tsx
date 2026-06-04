import React from "react";
import { X } from "lucide-react";

type DocsOverlayProps = {
  open: boolean;
  onClose: () => void;
};

export function DocsOverlay({ open, onClose }: DocsOverlayProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="docs-overlay" role="dialog" aria-label="API documentation">
      <div className="docs-overlay-toolbar">
        <strong>API Docs</strong>
        <button type="button" className="docs-overlay-close" onClick={onClose} aria-label="Close API docs">
          <X size={18} />
          Close
        </button>
      </div>
      <iframe className="docs-overlay-frame" title="FreeRouter API documentation" src="/docs" />
    </div>
  );
}
