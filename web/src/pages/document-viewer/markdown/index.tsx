import { Authorization } from '@/constants/authorization';
import { getAuthorization } from '@/utils/authorization-util';
import { Skeleton } from 'antd';
import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import FileError from '../file-error';
import { useCatchError } from '../hooks';
import styles from './index.less';

interface IProps {
  url: string;
}

const MarkdownPreviewer = ({ url }: IProps) => {
  const { error } = useCatchError(url);
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchMarkdown = async () => {
      const headers = {
        [Authorization]: getAuthorization(),
      };
      try {
        const response = await fetch(url, { headers });
        const text = await response.text();
        setContent(text);
      } catch (err) {
        console.error('Failed to fetch markdown:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMarkdown();
  }, [url]);

  if (error) {
    return <FileError>{error}</FileError>;
  }

  if (loading) {
    return <Skeleton active />;
  }

  return (
    <div className={styles.markdownContainer}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                language={match[1]}
                PreTag="div"
                wrapLongLines
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownPreviewer;
