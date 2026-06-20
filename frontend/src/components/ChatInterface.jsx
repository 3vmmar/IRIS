import React, { useEffect, useRef } from 'react';
import { User, Bot, Loader2 } from 'lucide-react';

export default function ChatInterface({ history, isTyping }) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [history, isTyping]);

  return (
    <div className="flex flex-col h-full bg-panel rounded-lg border border-border overflow-hidden">
      <div className="p-4 border-b border-border bg-surface font-medium text-sm text-textMain flex items-center gap-2">
        <Bot size={16} className="text-primary" />
        Conversation Context
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {history.length === 0 && !isTyping && (
          <div className="text-textMuted text-sm text-center mt-10">
            No messages yet. Say something to start!
          </div>
        )}
        
        {history.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0 text-primary">
                <Bot size={16} />
              </div>
            )}
            
            <div className={`px-4 py-2 rounded-2xl max-w-[80%] text-sm ${
              msg.role === 'user' 
                ? 'bg-primary text-white rounded-br-none' 
                : 'bg-surface text-textMain rounded-bl-none border border-border'
            }`}>
              {msg.content}
              {msg.isStreaming && <span className="inline-block w-1.5 h-4 ml-1 bg-primary animate-pulse align-middle"></span>}
            </div>

            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-surface border border-border flex items-center justify-center flex-shrink-0 text-textMuted">
                <User size={16} />
              </div>
            )}
          </div>
        ))}
        
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
