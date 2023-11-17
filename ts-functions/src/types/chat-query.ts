export interface ChatQuery {
    context: string;
    history?: {
        author: string;
        content?: string | null;
    }[]
}
