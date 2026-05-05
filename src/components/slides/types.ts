export interface Slide {
  id: string;
  variant: "divider" | "content" | "diagram";
  title: string;
  subtitle?: string;
  eyebrow?: string;
  content?: string;
}
