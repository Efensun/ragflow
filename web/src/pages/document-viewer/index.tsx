import { Images } from '@/constants/common';
import { api_host } from '@/utils/api';
import { previewHtmlFile } from '@/utils/file-util';
import { Flex } from 'antd';
import { useParams, useSearchParams } from 'umi';
import Docx from './docx';
import Excel from './excel';
import Image from './image';
import styles from './index.less';
import Markdown from './markdown';
import Pdf from './pdf';

// TODO: The interface returns an incorrect content-type for the SVG.

const DocumentViewer = () => {
  const { id: documentId } = useParams();
  const [currentQueryParameters] = useSearchParams();
  const ext = currentQueryParameters.get('ext');
  const prefix = currentQueryParameters.get('prefix');
  const api = `${api_host}/${prefix || 'file'}/get/${documentId}`;

  if (ext === 'html' && documentId) {
    previewHtmlFile(documentId);
    return;
  }

  return (
    <section className={styles.viewerWrapper}>
      {Images.includes(ext!) && (
        <Flex className={styles.image} align="center" justify="center">
          <Image src={api} preview={false}></Image>
        </Flex>
      )}
      {ext === 'pdf' && <Pdf url={api}></Pdf>}
      {(ext === 'xlsx' || ext === 'xls') && <Excel filePath={api}></Excel>}
      {ext === 'docx' && <Docx filePath={api}></Docx>}
      {ext === 'md' && <Markdown url={api}></Markdown>}
    </section>
  );
};

export default DocumentViewer;
