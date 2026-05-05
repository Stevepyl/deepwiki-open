import type { ButtonHTMLAttributes, ReactNode } from "react";

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
}

export function IconButton({ children, className = "", type = "button", ...props }: IconButtonProps) {
  return (
    <button
      className={`flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] text-[var(--ink-muted)] transition-all duration-[120ms] hover:bg-[var(--paper-panel)] hover:text-[var(--ink-primary)] [&>svg]:h-[15px] [&>svg]:w-[15px] ${className}`}
      type={type}
      {...props}
    >
      {children}
    </button>
  );
}
